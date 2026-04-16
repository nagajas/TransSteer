"""
transform.py — CPU-only bridge transformation.

Pipeline:
  1. Load source_anchors.npy  (N, src_dim)  
  2. Load target_anchors.npy  (N, tgt_dim)
  3. PCA-reduce source anchors to tgt_dim.
  4. Orthogonal Procrustes on (pca_source, target) → rotation R and scale s.
  5. Load the phase-1 steering vector dict, pick the requested --layer.
  6. Transform: v_transformed = pca.transform(v) @ R * s
  7. Save transformed_vector.npy (shape: (tgt_dim,)).
  8. Optionally save bridge.pkl with {pca, R, scale} for later reuse.
"""

import argparse
import pickle
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes


def load_anchors(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    assert arr.ndim == 2, f"Expected 2-D anchor matrix, got shape {arr.shape}"
    return arr.astype(np.float64)


def load_steering_vector(path: str, layer: int) -> np.ndarray:
    """Load the phase-1 .npy file (dict of {layer: ndarray}) and return the vector for `layer`."""
    data = np.load(path, allow_pickle=True).item()
    if layer not in data:
        available = sorted(data.keys())
        raise KeyError(f"Layer {layer} not found in steering vector file. Available layers: {available}")
    vec = data[layer].astype(np.float64)
    assert vec.ndim == 1, f"Expected 1-D steering vector for layer {layer}, got shape {vec.shape}"
    return vec


def fit_bridge(
    source_anchors: np.ndarray,
    target_anchors: np.ndarray,
) -> tuple[PCA, np.ndarray, float]:
    """
    Fit a PCA + Procrustes bridge from source to target space.

    Returns
    -------
    pca : fitted sklearn PCA (src_dim → tgt_dim)
    R   : orthogonal rotation matrix (tgt_dim × tgt_dim)
    s   : scalar scale factor
    """
    n_src, src_dim = source_anchors.shape
    n_tgt, tgt_dim = target_anchors.shape

    if n_src != n_tgt:
        raise ValueError(
            f"Anchor count mismatch: source has {n_src} rows, target has {n_tgt} rows. "
            "Both must be generated from the same wikitext samples."
        )

    if src_dim < tgt_dim:
        raise ValueError(
            f"Source hidden_dim ({src_dim}) is smaller than target hidden_dim ({tgt_dim}). "
            "Cannot reduce source to target dimensionality via PCA."
        )

    # --- Step 1: PCA reduce source to target dimensionality ---
    print(f"Fitting PCA: {src_dim}D → {tgt_dim}D on {n_src} anchor samples ...")
    pca = PCA(n_components=tgt_dim, random_state=42)
    source_reduced = pca.fit_transform(source_anchors)  # (N, tgt_dim)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance retained: {explained * 100:.2f}%")

    # --- Step 2: Center both sets (standard Procrustes pre-processing) ---
    src_mean = source_reduced.mean(axis=0)
    tgt_mean = target_anchors.mean(axis=0)
    A = source_reduced - src_mean
    B = target_anchors - tgt_mean

    # --- Step 3: Orthogonal Procrustes: find R s.t. A @ R ≈ B ---
    # scipy returns (R, scale) where scale = trace(A.T @ B @ R.T),
    # the numerator of the optimal isotropic scale factor.
    print("Fitting Orthogonal Procrustes ...")
    R, procrustes_scale = orthogonal_procrustes(A, B)

    # Isotropic scale: s = trace(A.T @ B @ R.T) / ||A||_F^2
    denom = np.sum(A ** 2)
    s = procrustes_scale / denom if denom > 0 else 1.0
    print(f"  Procrustes scale factor s = {s:.6f}")

    return pca, src_mean, tgt_mean, R, s


def transform_vector(
    vec: np.ndarray,
    pca: PCA,
    src_mean: np.ndarray,
    tgt_mean: np.ndarray,
    R: np.ndarray,
    s: float,
) -> np.ndarray:
    """
    Apply the bridge to a single steering vector.

    Steps mirror the anchor pre-processing:
      1. PCA project  (src_dim → tgt_dim)
      2. Centre in the PCA space (subtract src_mean that was used during Procrustes)
      3. Rotate and scale
      4. Re-add target mean so the vector lives in the same affine subspace as the target anchors
    """
    v_reduced = pca.transform(vec.reshape(1, -1))[0]   # (tgt_dim,)
    v_centred = v_reduced - src_mean                    # centre as during Procrustes
    v_rotated = v_centred @ R                           # rotate
    v_scaled  = v_rotated * s                           # isotropic scale
    v_final   = v_scaled + tgt_mean                     # un-centre into target space
    return v_final


def save_bridge(path: str, pca: PCA, src_mean: np.ndarray, tgt_mean: np.ndarray, R: np.ndarray, s: float) -> None:
    bridge = {"pca": pca, "src_mean": src_mean, "tgt_mean": tgt_mean, "R": R, "scale": s}
    with open(path, "wb") as f:
        pickle.dump(bridge, f)
    print(f"Saved bridge to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform a phase-1 steering vector from source model space to target model space."
    )
    parser.add_argument(
        "--source_anchors",
        default="./source_anchors.npy",
        help="Path to source model anchor activations (N, src_dim) produced by phase2a.",
    )
    parser.add_argument(
        "--target_anchors",
        default="./target_anchors.npy",
        help="Path to target model anchor activations (N, tgt_dim) produced by phase2a.",
    )
    parser.add_argument(
        "--steering_vec",
        default="./steering_vector.npy",
        help="Path to phase-1 steering vector file (dict of {layer: ndarray}).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer key to extract from the steering vector dict.",
    )
    parser.add_argument(
        "--out_vec",
        default="./transformed_vector.npy",
        help="Output path for the transformed steering vector (.npy).",
    )
    parser.add_argument(
        "--bridge_out",
        default=None,
        help="Optional output path for bridge.pkl (PCA model + R + scale) for reuse.",
    )
    args = parser.parse_args()

    # --- Load anchors ---
    print(f"Loading source anchors from {args.source_anchors} ...")
    source_anchors = load_anchors(args.source_anchors)
    print(f"  Shape: {source_anchors.shape}")

    print(f"Loading target anchors from {args.target_anchors} ...")
    target_anchors = load_anchors(args.target_anchors)
    print(f"  Shape: {target_anchors.shape}")

    # --- Fit bridge ---
    pca, src_mean, tgt_mean, R, s = fit_bridge(source_anchors, target_anchors)

    # --- Load steering vector ---
    print(f"Loading steering vector from {args.steering_vec} (layer {args.layer}) ...")
    steering_vec = load_steering_vector(args.steering_vec, args.layer)
    print(f"  Steering vector shape: {steering_vec.shape}")

    # Sanity check: steering vector dim must match source anchor dim
    src_dim = source_anchors.shape[1]
    if steering_vec.shape[0] != src_dim:
        raise ValueError(
            f"Steering vector dim ({steering_vec.shape[0]}) does not match "
            f"source anchor dim ({src_dim}). Ensure --layer matches the layer "
            "used when running phase2a on the source model."
        )

    # --- Transform ---
    print("Transforming steering vector ...")
    transformed = transform_vector(steering_vec, pca, src_mean, tgt_mean, R, s)
    print(f"  Transformed vector shape: {transformed.shape}")

    # --- Save ---
    np.save(args.out_vec, transformed.astype(np.float32))
    print(f"Saved transformed vector to {args.out_vec}")

    if args.bridge_out:
        save_bridge(args.bridge_out, pca, src_mean, tgt_mean, R, s)


if __name__ == "__main__":
    main()