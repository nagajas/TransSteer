"""
pipeline.py — End-to-end Transfer Steering orchestration.

Execution order (memory-safe, stays under 4 GB VRAM):

1. Load source model (Llama-2-7B, 4-bit quantised).
2. Extract steering vectors → save to ``vectors/``.
3. Unload source model completely.
4. Transform vectors to target model's space.
5. Load target model (SmolLM2-135M).
6. Apply steering vectors via forward hooks.
7. Evaluate baseline vs. steered model.
8. Save ``results.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from evaluation import calculate_error_rates, compute_alignment_score, evaluate_model
from extraction import VectorExtractor
from transformation import VectorTransformer
from utils import DatasetLoader, load_model, unload_model

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SOURCE_MODEL = "meta-llama/Llama-2-7b-hf"
TARGET_MODEL = "HuggingFaceTB/SmolLM2-135M"

SOURCE_NUM_LAYERS = 32
TARGET_NUM_LAYERS = 24  # fallback; actual value is always read from the target config
SOURCE_DIM = 4096
TARGET_DIM = 576  # SmolLM2-135M hidden dim

VECTORS_DIR = "vectors"
RESULTS_FILE = "results.json"

# ---------------------------------------------------------------------------
# Hook injection
# ---------------------------------------------------------------------------


class SteeringHookManager:
    """Inject pre-computed steering vectors into specific transformer layers.

    Uses PyTorch forward hooks to add the steering vector to the hidden states
    produced by a given module (typically a transformer block).

    Parameters
    ----------
    model:
        The target model into whose layers the hooks will be registered.
    layer_vector_map:
        ``{target_layer_index: steering_vector_1d}`` mapping.
    alpha:
        Scaling factor applied to the steering vector before addition.
        Larger values → stronger steering.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_vector_map: Dict[int, np.ndarray],
        alpha: float = 1.0,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        # Pre-convert vectors to tensors for efficiency
        self._tensors: Dict[int, torch.Tensor] = {
            idx: torch.from_numpy(vec).float()
            for idx, vec in layer_vector_map.items()
        }

    def _make_hook(self, vec: torch.Tensor):
        def hook(module, inputs, output):
            # output may be a tuple (hidden_state, ...) or just a tensor
            if isinstance(output, tuple):
                hs = output[0]
                steered = hs + vec.to(hs.device).to(hs.dtype) * self.alpha
                return (steered,) + output[1:]
            else:
                return output + vec.to(output.device).to(output.dtype) * self.alpha
        return hook

    def register(self) -> None:
        """Register all steering hooks on the model's transformer layers."""
        # Navigate to the layers list; works for most HuggingFace architectures.
        layers = _get_transformer_layers(self.model)
        for idx, vec in self._tensors.items():
            if idx < len(layers):
                handle = layers[idx].register_forward_hook(self._make_hook(vec))
                self._handles.append(handle)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Heuristically locate the main transformer block list in a model."""
    # Try common attribute paths used by different architectures
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        found = True
        for part in attr.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise AttributeError(
        "Could not locate transformer layers in the model. "
        "Please inspect model.named_modules() and update _get_transformer_layers()."
    )


# ---------------------------------------------------------------------------
# Helper: build prompt pairs from dataset records
# ---------------------------------------------------------------------------


def build_prompt_pairs(records: List[Dict]) -> List[Tuple[str, str]]:
    """Convert dataset records to ``(positive, negative)`` prompt pairs.

    Expected record schema (JSONL):
    ::

        {
          "positive": "...",
          "negative": "...",
          "prompt": "...",
          "choices": ["...", "..."],
          "correct": 0
        }

    If a record has ``"positive"`` and ``"negative"`` keys, those are used
    directly.  Otherwise a best-effort fallback is applied.
    """
    pairs = []
    for rec in records:
        pos = rec.get("positive") or rec.get("prompt", "")
        neg = rec.get("negative") or rec.get("prompt", "I disagree. ")
        pairs.append((pos, neg))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    source_model_name: str = SOURCE_MODEL,
    target_model_name: str = TARGET_MODEL,
    dataset_path: str = "data/sample.jsonl",
    method: str = "standard",
    dim_method: str = "pca",
    alpha: float = 1.0,
    clip_value: float = 1.0,
    noise_multiplier: float = 1.1,
    vectors_dir: str = VECTORS_DIR,
    results_file: str = RESULTS_FILE,
) -> Dict:
    """Run the full Transfer Steering pipeline.

    Parameters
    ----------
    source_model_name:
        HuggingFace identifier for the source model.
    target_model_name:
        HuggingFace identifier for the target model.
    dataset_path:
        Path to the JSONL dataset file.
    method:
        Extraction method: ``"standard"`` or ``"private"``.
    dim_method:
        Dimension alignment method: ``"pca"`` or ``"linear"``.
    alpha:
        Steering strength multiplier.
    clip_value:
        DP clip value (only used when ``method="private"``).
    noise_multiplier:
        DP noise multiplier (only used when ``method="private"``).
    vectors_dir:
        Directory to save/load steering vectors.
    results_file:
        Output JSON file path.

    Returns
    -------
    Dict with alignment scores and error rates.
    """
    os.makedirs(vectors_dir, exist_ok=True)
    dataset = DatasetLoader(dataset_path)
    train_records = dataset.train
    test_records = dataset.test

    print(f"\n{'='*60}")
    print(f"TransSteer Pipeline")
    print(f"  Source model : {source_model_name}")
    print(f"  Target model : {target_model_name}")
    print(f"  Dataset      : {dataset_path} ({len(dataset)} records)")
    print(f"  Method       : {method} | dim_method={dim_method}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Step 1: Extract steering vectors from the source model
    # ------------------------------------------------------------------
    print("[Step 1] Loading source model …")
    source_model, source_tokenizer = load_model(source_model_name)

    src_num_layers = source_model.config.num_hidden_layers
    src_dim = source_model.config.hidden_size

    extractor = VectorExtractor(
        model=source_model,
        tokenizer=source_tokenizer,
        vectors_dir=vectors_dir,
    )

    pairs = build_prompt_pairs(train_records)
    print(f"[Step 2] Extracting vectors ({method}) from {len(pairs)} pairs …")
    if method == "private":
        vector_paths = extractor.extract_private(
            pairs,
            clip_value=clip_value,
            noise_multiplier=noise_multiplier,
            tag=method,
        )
    else:
        vector_paths = extractor.extract_standard(pairs, tag=method)

    # ------------------------------------------------------------------
    # Step 2: Unload source model
    # ------------------------------------------------------------------
    print("[Step 3] Unloading source model from VRAM …")
    extractor.unload()  # calls unload_model internally

    # ------------------------------------------------------------------
    # Step 3: Transform vectors
    # ------------------------------------------------------------------
    print("[Step 4] Transforming vectors …")

    # Load target model config to get actual dimensions (without loading weights)
    from transformers import AutoConfig
    tgt_config = AutoConfig.from_pretrained(target_model_name)
    tgt_num_layers = tgt_config.num_hidden_layers
    tgt_dim = tgt_config.hidden_size

    transformer = VectorTransformer(
        source_num_layers=src_num_layers,
        target_num_layers=tgt_num_layers,
        source_dim=src_dim,
        target_dim=tgt_dim,
        method=dim_method,
    )
    transformed = transformer.load_and_transform(vectors_dir, tag=method)
    transformer.save_transformed(transformed, vectors_dir, tag="transformed")

    # Build {target_layer_index: vector} map for hook injection
    tgt_layer_vector_map: Dict[int, np.ndarray] = {
        tgt_idx: vec for _src, (tgt_idx, vec) in transformed.items()
    }

    # ------------------------------------------------------------------
    # Step 4: Load target model
    # ------------------------------------------------------------------
    print("[Step 5] Loading target model …")
    target_model, target_tokenizer = load_model(target_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Step 5: Baseline evaluation (no steering)
    # ------------------------------------------------------------------
    print("[Step 6a] Baseline evaluation (no steering) …")
    baseline_train = evaluate_model(
        target_model, target_tokenizer, train_records, device=device
    )
    baseline_test = evaluate_model(
        target_model, target_tokenizer, test_records, device=device
    )

    # ------------------------------------------------------------------
    # Step 6: Steered evaluation
    # ------------------------------------------------------------------
    print("[Step 6b] Steered evaluation …")
    hook_manager = SteeringHookManager(target_model, tgt_layer_vector_map, alpha=alpha)
    hook_manager.register()

    steered_train = evaluate_model(
        target_model, target_tokenizer, train_records, device=device
    )
    steered_test = evaluate_model(
        target_model, target_tokenizer, test_records, device=device
    )
    hook_manager.remove()

    # ------------------------------------------------------------------
    # Step 7: Compute metrics
    # ------------------------------------------------------------------
    print("[Step 7] Computing metrics …")

    def best_log_prob(results: List[Dict]) -> List[float]:
        return [max(r["log_probs"]) if r["log_probs"] else float("-inf") for r in results]

    baseline_member_scores = best_log_prob(baseline_train)
    baseline_non_member_scores = best_log_prob(baseline_test)
    steered_member_scores = best_log_prob(steered_train)
    steered_non_member_scores = best_log_prob(steered_test)

    baseline_error_rates = calculate_error_rates(
        baseline_member_scores, baseline_non_member_scores
    )
    steered_error_rates = calculate_error_rates(
        steered_member_scores, steered_non_member_scores
    )

    alignment_score = compute_alignment_score(baseline_train + baseline_test, steered_train + steered_test)

    results = {
        "source_model": source_model_name,
        "target_model": target_model_name,
        "method": method,
        "dim_method": dim_method,
        "alpha": alpha,
        "alignment_score": alignment_score,
        "baseline_error_rates": baseline_error_rates,
        "steered_error_rates": steered_error_rates,
    }

    # ------------------------------------------------------------------
    # Step 8: Save results
    # ------------------------------------------------------------------
    with open(results_file, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[Done] Results saved to {results_file}")
    print(json.dumps(results, indent=2))

    unload_model(target_model)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TransSteer Transfer Steering Pipeline")
    parser.add_argument("--source-model", default=SOURCE_MODEL)
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--dataset", default="data/sample.jsonl")
    parser.add_argument("--method", choices=["standard", "private"], default="standard")
    parser.add_argument("--dim-method", choices=["pca", "linear"], default="pca")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clip-value", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--vectors-dir", default=VECTORS_DIR)
    parser.add_argument("--results-file", default=RESULTS_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        source_model_name=args.source_model,
        target_model_name=args.target_model,
        dataset_path=args.dataset,
        method=args.method,
        dim_method=args.dim_method,
        alpha=args.alpha,
        clip_value=args.clip_value,
        noise_multiplier=args.noise_multiplier,
        vectors_dir=args.vectors_dir,
        results_file=args.results_file,
    )
