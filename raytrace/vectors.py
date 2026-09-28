from __future__ import annotations

import numpy as np

EPS = 1.0e-10


def as_vec(value) -> np.ndarray:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,):
        raise ValueError("a three-component vector is required")
    return vec


def normalized(value) -> np.ndarray:
    vec = as_vec(value)
    norm = float(np.linalg.norm(vec))
    if norm <= EPS:
        raise ValueError("cannot normalize a zero vector")
    return vec / norm


def dot(a, b) -> float:
    return float(np.dot(as_vec(a), as_vec(b)))

