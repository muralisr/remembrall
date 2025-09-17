from __future__ import annotations
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    # Euclidean distances
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt((diffs**2).sum(axis=-1) + 1e-8)

def pca_2d(X: np.ndarray):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T
