"""
extraction.py — Steering vector extraction from a source model.

Two extraction methods are supported:

* **Standard Mean** – plain arithmetic mean of the difference between
  "Positive" and "Negative" prompt activations.
* **Private Mean** – differentially-private variant that clips each
  per-sample gradient and adds calibrated Gaussian noise.

Vectors are saved as ``.npy`` files inside a ``vectors/`` directory and the
model is fully unloaded from VRAM after extraction.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import unload_model

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_VECTORS_DIR = "vectors"


def _get_layer_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_indices: List[int],
    device: str,
) -> List[np.ndarray]:
    """Return the mean-pooled hidden state at each requested *layer_indices*."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple of (num_layers+1) tensors, each [1, seq, hidden]
    hidden_states = outputs.hidden_states  # includes embedding layer at index 0
    results = []
    for idx in layer_indices:
        # Mean-pool over the sequence dimension → shape [hidden]
        hs = hidden_states[idx].squeeze(0).mean(dim=0)
        results.append(hs.float().cpu().numpy())
    return results


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class VectorExtractor:
    """Extract steering vectors from a loaded source model.

    The extractor iterates over pairs of ``(positive_prompt, negative_prompt)``
    and computes the activation difference at specific transformer layers.

    Parameters
    ----------
    model:
        A loaded ``AutoModelForCausalLM`` instance (e.g. Llama-2-7B-hf).
    tokenizer:
        Matching tokeniser.
    layer_indices:
        Which hidden-state layers to extract from.  Index 0 is the embedding
        layer; indices 1…N correspond to transformer blocks 0…N-1.
    vectors_dir:
        Directory where ``.npy`` files will be saved.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer_indices: Optional[List[int]] = None,
        vectors_dir: str = _DEFAULT_VECTORS_DIR,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vectors_dir = vectors_dir
        os.makedirs(vectors_dir, exist_ok=True)

        # Infer default layer range: roughly layers 17-21 for a 32-layer model
        num_layers = model.config.num_hidden_layers
        if layer_indices is None:
            start = int(num_layers * 0.50)
            end = int(num_layers * 0.65)
            layer_indices = list(range(start, end + 1))
        # Hidden-state index 0 is embedding; block i → index i+1
        self.layer_indices: List[int] = [i + 1 for i in layer_indices]

        # Determine device (handles 4-bit models spread across devices)
        self._device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_differences(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[np.ndarray]:
        """Return per-layer mean difference vectors (Standard Mean)."""
        # Accumulate differences: list-of-layers, each holding list-of-samples
        layer_diffs: List[List[np.ndarray]] = [[] for _ in self.layer_indices]

        for pos_prompt, neg_prompt in tqdm(pairs, desc="Extracting activations"):
            pos_hs = _get_layer_hidden_state(
                self.model, self.tokenizer, pos_prompt,
                self.layer_indices, str(self._device)
            )
            neg_hs = _get_layer_hidden_state(
                self.model, self.tokenizer, neg_prompt,
                self.layer_indices, str(self._device)
            )
            for li, (p, n) in enumerate(zip(pos_hs, neg_hs)):
                layer_diffs[li].append(p - n)

        # Average over all samples
        return [np.mean(np.stack(diffs, axis=0), axis=0) for diffs in layer_diffs]

    def _compute_private_differences(
        self,
        pairs: List[Tuple[str, str]],
        clip_value: float,
        noise_multiplier: float,
    ) -> List[np.ndarray]:
        """Return per-layer differentially-private mean difference vectors.

        Each per-sample difference vector is clipped to ``clip_value`` L2-norm
        and then Gaussian noise scaled by ``noise_multiplier * clip_value / n``
        is added to the aggregate, following the standard DP-SGD recipe.
        """
        layer_diffs: List[List[np.ndarray]] = [[] for _ in self.layer_indices]

        for pos_prompt, neg_prompt in tqdm(pairs, desc="Extracting (DP) activations"):
            pos_hs = _get_layer_hidden_state(
                self.model, self.tokenizer, pos_prompt,
                self.layer_indices, str(self._device)
            )
            neg_hs = _get_layer_hidden_state(
                self.model, self.tokenizer, neg_prompt,
                self.layer_indices, str(self._device)
            )
            for li, (p, n) in enumerate(zip(pos_hs, neg_hs)):
                diff = p - n
                # Clip to bound the L2 sensitivity
                norm = np.linalg.norm(diff)
                if norm > clip_value:
                    diff = diff * (clip_value / norm)
                layer_diffs[li].append(diff)

        n = len(pairs)
        result = []
        for diffs in layer_diffs:
            agg = np.sum(np.stack(diffs, axis=0), axis=0)
            # Gaussian noise with std = noise_multiplier * clip_value
            noise_std = noise_multiplier * clip_value
            noise = np.random.normal(0.0, noise_std, size=agg.shape)
            private_mean = (agg + noise) / n
            result.append(private_mean)
        return result

    def _save_vectors(
        self,
        vectors: List[np.ndarray],
        tag: str,
    ) -> List[str]:
        """Persist *vectors* to disk and return the saved file paths."""
        paths = []
        for li, vec in zip(self.layer_indices, vectors):
            fname = os.path.join(self.vectors_dir, f"{tag}_layer{li}.npy")
            np.save(fname, vec)
            paths.append(fname)
        return paths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_standard(
        self,
        pairs: List[Tuple[str, str]],
        tag: str = "standard",
    ) -> List[str]:
        """Extract vectors using the plain arithmetic mean.

        Parameters
        ----------
        pairs:
            List of ``(positive_prompt, negative_prompt)`` string tuples.
        tag:
            Filename prefix used when saving ``.npy`` files.

        Returns
        -------
        List of file paths for the saved vectors.
        """
        vectors = self._compute_differences(pairs)
        paths = self._save_vectors(vectors, tag)
        print(f"[extraction] Standard vectors saved to: {paths}")
        return paths

    def extract_private(
        self,
        pairs: List[Tuple[str, str]],
        clip_value: float = 1.0,
        noise_multiplier: float = 1.1,
        tag: str = "private",
    ) -> List[str]:
        """Extract vectors using a differentially-private mean.

        Parameters
        ----------
        pairs:
            List of ``(positive_prompt, negative_prompt)`` string tuples.
        clip_value:
            L2 sensitivity bound applied to each per-sample difference.
        noise_multiplier:
            Controls the amount of Gaussian noise added (higher → more privacy,
            less accuracy).
        tag:
            Filename prefix used when saving ``.npy`` files.

        Returns
        -------
        List of file paths for the saved vectors.
        """
        vectors = self._compute_private_differences(pairs, clip_value, noise_multiplier)
        paths = self._save_vectors(vectors, tag)
        print(f"[extraction] Private (DP) vectors saved to: {paths}")
        return paths

    def unload(self) -> None:
        """Unload the source model from VRAM.

        Must be called after extraction is complete so that the target model
        can fit inside the 4 GB GPU budget.
        """
        unload_model(self.model)
        self.model = None  # type: ignore[assignment]
        print("[extraction] Source model unloaded from VRAM.")
