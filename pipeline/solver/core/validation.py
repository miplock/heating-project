"""Validation helpers for solver inputs."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pipeline.solver.types import Array


def validate_u(u: Array, shape: Tuple[int, int]) -> Array:
    """Validate input array shape and values."""
    # Typ i ksztalt tablicy.
    if not isinstance(u, np.ndarray):
        raise TypeError("u must be np.ndarray.")
    if u.ndim != 2:
        raise ValueError("u must be a 2D array (ny, nx).")
    if u.shape != shape:
        msg = f"Wrong u shape: {u.shape}, expected {shape}."
        raise ValueError(msg)
    if not np.isfinite(u).all():
        raise ValueError("u contains NaN/Inf.")
    return u
