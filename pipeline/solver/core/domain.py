"""Domain mask utilities."""

from __future__ import annotations

from pipeline.solver.types import Array, Masks


def apply_domain_mask(unew: Array, u_prev: Array, masks: Masks) -> Array:
    """Clamp updates outside the domain to the previous values."""
    # Maska domeny, jesli kiedys bedzie nieregularna.
    dom = masks.domain
    if dom is None:
        return unew

    # Poza domena zachowujemy poprzednie wartosci.
    out = unew.copy()
    out[~dom] = u_prev[~dom]
    return out
