"""Heat source utilities for the radiator."""

from __future__ import annotations

import numpy as np

from pipeline.solver.types import Array, Masks, RadiatorParams


def source_term(
    u: Array,
    *,
    masks: Masks,
    radiator: RadiatorParams,
    air_cp: float,
    air_r: float,
    air_p: float,
) -> Array:
    """Compute the heat source term for the radiator."""
    # Jezeli grzejnik wylaczony, brak zrodla.
    if not radiator.enabled:
        return np.zeros_like(u)

    # Maski grzejnika.
    rad = masks.radiator
    if rad is None:
        return np.zeros_like(u)

    # Srednia temperatura w pokoju (calosc lub maska).
    dom = masks.domain
    if dom is None:
        mean_u = float(u.mean())
    else:
        mean_u = float(u[dom].mean())

    # Jesli jest juz cieplo, grzejnik nic nie robi.
    if mean_u >= radiator.setpoint:
        return np.zeros_like(u)

    # Gestosc z rownania gazu doskonalego: rho = p / (r * T).
    rho = air_p / (air_r * u)
    # Zrodlo ciepla tylko na masce grzejnika.
    src = np.zeros_like(u)
    # Przeliczenie mocy objetosciowej na zmiane temperatury.
    src[rad] = radiator.coeff * u[rad] / (rho[rad] * air_cp)
    return src
