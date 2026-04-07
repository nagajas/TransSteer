"""
transformation.py — Dimensionality alignment and layer mapping for steering vectors.

The **VectorTransformer** class bridges the gap between a large source model
(e.g. Llama-2-7B, hidden-dim 4096, 32 layers) and a smaller target model
(e.g. SmolLM2-135M, hidden-dim 576, 24 layers) by:

1. **Layer Mapping** – maps source layers expressed as a fraction of the total
   source depth to the corresponding fraction of the target depth.
2. **Dimension Alignment** – shrinks the 4096-d source vector to 576-d using
   either PCA (unsupervised) or a trainable Linear Projection.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------

def map_source_to_target_layers(
    source_layer_indices: List[int],
    source_num_layers: int,
    target_num_layers: int,
) -> Dict[int, int]:
    """Map a list of source layer indices to their target-model equivalents.

    The mapping preserves the *relative* position of each layer within its
    model.  For example, layer 17 of 32 (≈ 53 %) maps to layer 6 of 12
    (≈ 53 % → 6.4 → 6).

    Parameters
    ----------
    source_layer_indices:
        Absolute layer indices in the source model (0-based, counting from the
        first transformer block, not the embedding layer).
    source_num_layers:
        Total number of transformer blocks in the source model.
    target_num_layers:
        Total number of transformer blocks in the target model.

    Returns
    -------
    Dict mapping each source index to its corresponding target index.
    """
    mapping: Dict[int, int] = {}
    for src_idx in source_layer_indices:
        relative = src_idx / source_num_layers
        tgt_idx = min(
            int(round(relative * target_num_layers)),
            target_num_layers - 1,
        )
        mapping[src_idx] = tgt_idx
    return mapping


# ---------------------------------------------------------------------------
# Dimension alignment
# ---------------------------------------------------------------------------

class LinearProjection(nn.Module):
    """Learnable linear projection from *in_dim* to *out_dim*.

    The weight matrix is initialised with Xavier uniform and can be fine-tuned
    on labelled pairs if desired; by default it is used as a fixed random
    projection (which already preserves geometry approximately).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.proj(x)

    def project_numpy(self, vec: np.ndarray) -> np.ndarray:
        """Project a 1-D numpy vector and return a 1-D numpy array."""
        t = torch.from_numpy(vec).float().unsqueeze(0)
        with torch.no_grad():
            out = self.proj(t)
        return out.squeeze(0).numpy()


class PCAProjection:
    """PCA-based dimensionality reduction fitted on a set of source vectors.

    Parameters
    ----------
    target_dim:
        Number of principal components to retain (i.e. the target model's
        hidden dimension, e.g. 576).
    """

    def __init__(self, target_dim: int) -> None:
        self.target_dim = target_dim
        self._pca: Optional[PCA] = None

    def fit(self, vectors: np.ndarray) -> "PCAProjection":
        """Fit PCA on *vectors* (shape ``[n_samples, source_dim]``)."""
        n_components = min(self.target_dim, vectors.shape[0], vectors.shape[1])
        self._pca = PCA(n_components=n_components)
        self._pca.fit(vectors)
        return self

    def project(self, vec: np.ndarray) -> np.ndarray:
        """Project a 1-D vector down to ``target_dim`` dimensions.

        If the fitted number of components is less than *target_dim*, the
        output is zero-padded to reach the exact target dimensionality.
        """
        if self._pca is None:
            raise RuntimeError("PCAProjection has not been fitted yet.")
        projected = self._pca.transform(vec.reshape(1, -1)).squeeze(0)
        if projected.shape[0] < self.target_dim:
            pad = np.zeros(self.target_dim - projected.shape[0], dtype=projected.dtype)
            projected = np.concatenate([projected, pad])
        return projected


# ---------------------------------------------------------------------------
# VectorTransformer
# ---------------------------------------------------------------------------

class VectorTransformer:
    """Transform source steering vectors into the target model's space.

    Parameters
    ----------
    source_num_layers:
        Number of transformer blocks in the source model.
    target_num_layers:
        Number of transformer blocks in the target model.
    source_dim:
        Hidden dimension of the source model (e.g. 4096 for Llama-2-7B).
    target_dim:
        Hidden dimension of the target model (e.g. 576 for SmolLM2-135M).
    method:
        Dimension reduction strategy: ``"pca"`` (default) or ``"linear"``.
    """

    def __init__(
        self,
        source_num_layers: int,
        target_num_layers: int,
        source_dim: int,
        target_dim: int,
        method: str = "pca",
    ) -> None:
        self.source_num_layers = source_num_layers
        self.target_num_layers = target_num_layers
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.method = method.lower()

        if self.method == "pca":
            self._projector: PCAProjection | LinearProjection = PCAProjection(target_dim)
        elif self.method == "linear":
            self._projector = LinearProjection(source_dim, target_dim)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'pca' or 'linear'.")

    # ------------------------------------------------------------------
    # Fitting (PCA only)
    # ------------------------------------------------------------------

    def fit(self, source_vectors: np.ndarray) -> "VectorTransformer":
        """Fit the PCA projector on *source_vectors*.

        Only required when ``method="pca"``.  Linear projection requires no
        fitting (it uses Xavier-initialised weights).

        Parameters
        ----------
        source_vectors:
            Array of shape ``[n_samples, source_dim]``.
        """
        if self.method == "pca":
            self._projector.fit(source_vectors)  # type: ignore[attr-defined]
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform_vectors(
        self,
        source_layer_map: Dict[int, np.ndarray],
    ) -> Dict[int, Tuple[int, np.ndarray]]:
        """Transform a dictionary of source vectors.

        Parameters
        ----------
        source_layer_map:
            ``{source_layer_index: vector_1d}`` mapping loaded from disk.

        Returns
        -------
        ``{source_layer_index: (target_layer_index, transformed_vector)}``
        """
        source_indices = list(source_layer_map.keys())
        layer_mapping = map_source_to_target_layers(
            source_indices, self.source_num_layers, self.target_num_layers
        )

        # For PCA: fit on all source vectors stacked together if not yet fitted
        if self.method == "pca" and self._projector._pca is None:  # type: ignore[union-attr]
            all_vecs = np.stack(list(source_layer_map.values()), axis=0)
            self.fit(all_vecs)

        result: Dict[int, Tuple[int, np.ndarray]] = {}
        for src_idx, vec in source_layer_map.items():
            tgt_idx = layer_mapping[src_idx]
            if self.method == "pca":
                transformed = self._projector.project(vec)  # type: ignore[union-attr]
            else:
                transformed = self._projector.project_numpy(vec)  # type: ignore[union-attr]
            result[src_idx] = (tgt_idx, transformed)
        return result

    # ------------------------------------------------------------------
    # Convenience: load from disk and transform
    # ------------------------------------------------------------------

    def load_and_transform(
        self,
        vectors_dir: str,
        tag: str,
    ) -> Dict[int, Tuple[int, np.ndarray]]:
        """Load ``.npy`` files from *vectors_dir* with filename prefix *tag*
        and transform them.

        Filenames must follow the convention produced by ``VectorExtractor``:
        ``<tag>_layer<N>.npy``.

        Returns
        -------
        Same structure as :meth:`transform_vectors`.
        """
        source_layer_map: Dict[int, np.ndarray] = {}
        for fname in sorted(os.listdir(vectors_dir)):
            if fname.startswith(tag) and fname.endswith(".npy"):
                layer_str = fname.replace(tag + "_layer", "").replace(".npy", "")
                try:
                    layer_idx = int(layer_str)
                except ValueError:
                    continue
                vec = np.load(os.path.join(vectors_dir, fname))
                # Convert from hidden-state index (1-based) to block index (0-based)
                source_layer_map[layer_idx - 1] = vec

        if not source_layer_map:
            raise FileNotFoundError(
                f"No vector files found for tag '{tag}' in '{vectors_dir}'."
            )

        return self.transform_vectors(source_layer_map)

    def save_transformed(
        self,
        transformed: Dict[int, Tuple[int, np.ndarray]],
        vectors_dir: str,
        tag: str = "transformed",
    ) -> List[str]:
        """Save transformed vectors to disk.

        Returns
        -------
        List of saved file paths.
        """
        os.makedirs(vectors_dir, exist_ok=True)
        paths = []
        for src_idx, (tgt_idx, vec) in transformed.items():
            fname = os.path.join(vectors_dir, f"{tag}_src{src_idx}_tgt{tgt_idx}.npy")
            np.save(fname, vec)
            paths.append(fname)
        print(f"[transformation] Transformed vectors saved: {paths}")
        return paths
