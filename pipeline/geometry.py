from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pipeline.solver import Grid, Masks


Array = np.ndarray


@dataclass(frozen=True)
class RoomGeometry:
    """
    Konfiguracja geometrii pokoju dla eksperymentu 1.

    - okno jest na gornej scianie (row=0)
    - r to odleglosc grzejnika od okna w osi y (w metrach),
      liczona od gornej sciany w dol
    """

    window_width: float
    radiator_size: Tuple[float, float]  # (width_x, height_y) w metrach
    radiator_offset_r: float  # odleglosc od okna w osi y [m]
    radiator_center_x: Optional[float] = None  # None -> srodek okna/pokoju


def build_room_masks(grid: Grid, geom: RoomGeometry) -> Masks:
    """
    Generator mask: sciany, okno i grzejnik dla prostokatnego pokoju.

    Zwraca Masks z polami: boundary, windows, walls, radiator, domain.
    """
    ny, nx = grid.shape
    hx = grid.hx

    boundary = np.zeros((ny, nx), dtype=bool)
    boundary[0, :] = True
    boundary[ny - 1, :] = True
    boundary[:, 0] = True
    boundary[:, nx - 1] = True

    window = np.zeros((ny, nx), dtype=bool)
    win_w_cells = _cells(geom.window_width, hx)
    win_center_x = _center_x_index(grid, geom.radiator_center_x)
    win_x0, win_x1 = _span_indices(win_center_x, win_w_cells, nx)
    window[0, win_x0:win_x1] = True

    walls = boundary & ~window

    radiator = np.zeros((ny, nx), dtype=bool)
    rad_w_cells = _cells(geom.radiator_size[0], hx)
    rad_h_cells = _cells(geom.radiator_size[1], hx)
    rad_center_x = win_center_x if geom.radiator_center_x is None else _x_to_index(
        geom.radiator_center_x, hx, nx
    )
    rad_x0, rad_x1 = _span_indices(rad_center_x, rad_w_cells, nx)
    rad_y0 = _y_to_index(geom.radiator_offset_r, hx, ny)
    rad_y1 = rad_y0 + rad_h_cells

    _validate_radiator_box(rad_x0, rad_x1, rad_y0, rad_y1, nx, ny)
    radiator[rad_y0:rad_y1, rad_x0:rad_x1] = True

    domain = np.ones((ny, nx), dtype=bool)

    return Masks(boundary=boundary, windows=window, walls=walls, radiator=radiator, domain=domain)


def _cells(length_m: float, hx: float) -> int:
    if length_m <= 0:
        raise ValueError("Dlugosc musi byc dodatnia.")
    return max(1, int(np.ceil(length_m / hx)))


def _center_x_index(grid: Grid, x_m: Optional[float]) -> int:
    if x_m is None:
        return (grid.nx - 1) // 2
    return _x_to_index(x_m, grid.hx, grid.nx)


def _x_to_index(x_m: float, hx: float, nx: int) -> int:
    idx = int(round(x_m / hx))
    return int(np.clip(idx, 0, nx - 1))


def _y_to_index(y_m: float, hx: float, ny: int) -> int:
    idx = int(round(y_m / hx))
    return int(np.clip(idx, 0, ny - 1))


def _span_indices(center: int, width_cells: int, n: int) -> Tuple[int, int]:
    half = width_cells // 2
    x0 = max(0, center - half)
    x1 = min(n, x0 + width_cells)
    x0 = max(0, x1 - width_cells)
    return x0, x1


def _validate_radiator_box(
    x0: int, x1: int, y0: int, y1: int, nx: int, ny: int
) -> None:
    if x0 <= 0 or x1 >= nx or y0 <= 0 or y1 >= ny:
        raise ValueError(
            "Grzejnik musi byc calkowicie wewnatrz domeny (bez brzegu)."
        )
