"""Small utilities used by the Experiment 1 runner."""

from __future__ import annotations

from typing import List

import numpy as np

from pipeline.solver import Grid


def grid_from_room(width_m: float, height_m: float, hx: float) -> Grid:
    if width_m <= 0 or height_m <= 0 or hx <= 0:
        raise ValueError("room size and hx must be positive.")
    nx = int(round(width_m / hx)) + 1
    ny = int(round(height_m / hx)) + 1
    return Grid(nx=nx, ny=ny, hx=hx)


def default_dt() -> float:
    return 0.01


def allowed_r_range(grid: Grid, rad_height_m: float) -> tuple[float, float]:
    rad_h_cells = max(1, int(np.ceil(rad_height_m / grid.hx)))
    min_r = grid.hx
    max_r = (grid.ny - 1 - rad_h_cells) * grid.hx
    if max_r < min_r:
        raise ValueError("Radiator height too large for the room height.")
    return min_r, max_r


def build_r_values(
    raw_values: str | None, r_min: float, r_max: float, r_count: int
) -> List[float]:
    if raw_values:
        values = [
            float(item.strip())
            for item in raw_values.split(",")
            if item.strip()
        ]
        if not values:
            raise ValueError("Empty --r-values list.")
        return values

    if r_count < 2:
        return [float(r_min)]
    return list(np.linspace(r_min, r_max, r_count))


def C_to_K(temp_c: float) -> float:
    return temp_c + 273.15
