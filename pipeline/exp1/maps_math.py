# Ten plik zawiera obliczenia i przygotowanie danych do map temperatury exp1.
"""Math and data-prep routines for exp1 temperature maps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from pipeline.exp1.common import C_to_K, Exp1VizConfig, grid_from_room
from pipeline.geometry import RoomGeometry, build_room_masks
from pipeline.solver import (
    BoundaryParams,
    HeatEquationSolver2D,
    PhysicalParams,
    RadiatorParams,
    TimeGrid,
)

OverlayPoints = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class MapInputs:
    """Bundle map data arrays and overlay coordinates for Plotly.

    Attributes:
        u_end_C: Final temperature field in Celsius.
        x: X coordinates for the grid.
        y: Y coordinates for the grid.
        overlays: Mapping of overlay name to x/y point arrays.
    """

    u_end_C: np.ndarray
    x: np.ndarray
    y: np.ndarray
    overlays: Dict[str, OverlayPoints]


def _mask_points(
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> OverlayPoints:
    """Extract x/y coordinates of True cells from a boolean mask.

    Args:
        mask: Boolean mask over the grid.
        x: X coordinates for the grid columns.
        y: Y coordinates for the grid rows.

    Returns:
        Tuple of (x_points, y_points). Arrays are empty if mask has no points.
    """
    points = np.argwhere(mask)
    if points.size == 0:
        return np.array([]), np.array([])
    ys = y[points[:, 0]]
    xs = x[points[:, 1]]
    return xs, ys


def simulate_end_state(config: Exp1VizConfig, r_m: float) -> np.ndarray:
    """Simulate exp1 and return the final temperature field in Kelvin.

    Args:
        config: Visualization/simulation configuration.
        r_m: Radiator offset in meters.

    Returns:
        2D array with temperatures in Kelvin at the final time step.
    """
    # Utworzenie siatki numerycznej na podstawie geometrii pokoju.
    grid = grid_from_room(config.room_width_m, config.room_height_m, config.hx)
    # Siatka czasowa od t=0 do t_end z krokiem dt.
    time = TimeGrid(dt=config.dt, t_end=config.t_end)
    # Parametry fizyczne dyfuzji ciepla.
    phys = PhysicalParams(alpha=config.alpha)
    # Parametry warunku brzegowego (dirichlet lub inny).
    bc = (
        BoundaryParams(kind="dirichlet", dirichlet_value=C_to_K(config.u_out_C))
        if config.bc_kind == "dirichlet"
        else BoundaryParams(kind=config.bc_kind)
    )

    # Geometria pomieszczenia i polozenie grzejnika.
    geom = RoomGeometry(
        window_width=config.window_width_m,
        radiator_size=config.radiator_size_m,
        radiator_offset_r=r_m,
        radiator_center_x=config.radiator_center_x_m,
    )
    # Maski geometrii: sciany, okna, grzejnik.
    masks = build_room_masks(grid, geom)
    # Parametry grzejnika i jego sterowania.
    radiator = RadiatorParams(
        enabled=True,
        setpoint=C_to_K(config.setpoint_C),
        coeff=config.radiator_coeff,
    )

    # Solver rownania ciepla w 2D.
    solver = HeatEquationSolver2D(
        grid=grid,
        time=time,
        phys=phys,
        bc=bc,
        masks=masks,
        radiator=radiator,
    )
    # Warunek poczatkowy w Kelvinach.
    u0 = np.full(grid.shape, C_to_K(config.u0_C))
    # Symulacja z zapisem tylko ostatniego kroku.
    out = solver.simulate(u0, store_every=time.n_steps)
    # Zwracamy pole temperatury z ostatniej chwili.
    return out["U"][-1]


def build_map_inputs(config: Exp1VizConfig, r_m: float) -> MapInputs:
    """Build the data needed to render a temperature map for one r.

    Args:
        config: Visualization/simulation configuration.
        r_m: Radiator offset in meters.

    Returns:
        MapInputs with temperature field, axes, and overlay coordinates.
    """
    # Siatka i geometria potrzebne do wyznaczenia masek.
    grid = grid_from_room(config.room_width_m, config.room_height_m, config.hx)
    # Parametry okna i grzejnika dla danego r.
    geom = RoomGeometry(
        window_width=config.window_width_m,
        radiator_size=config.radiator_size_m,
        radiator_offset_r=r_m,
        radiator_center_x=config.radiator_center_x_m,
    )
    # Maski do nakladek na wykres (sciana, okno, grzejnik).
    masks = build_room_masks(grid, geom)
    # Koncowa temperatura w stopniach Celsjusza.
    u_end_C = simulate_end_state(config, r_m) - 273.15

    # Wspolrzedne osi Y (wiersze siatki).
    y = np.linspace(0.0, config.room_height_m, u_end_C.shape[0])
    # Wspolrzedne osi X (kolumny siatki).
    x = np.linspace(0.0, config.room_width_m, u_end_C.shape[1])

    # Punkty masek jako zestaw wspolrzednych do rysowania.
    overlays = {
        "boundary": _mask_points(masks.boundary, x, y),
        "window": _mask_points(masks.windows, x, y),
        "radiator": _mask_points(masks.radiator, x, y),
    }
    return MapInputs(u_end_C=u_end_C, x=x, y=y, overlays=overlays)


def build_map_inputs_from_u_end(
    config: Exp1VizConfig,
    r_m: float,
    u_end_K: np.ndarray,
) -> MapInputs:
    """Build map inputs from a precomputed final temperature field."""
    grid = grid_from_room(config.room_width_m, config.room_height_m, config.hx)
    geom = RoomGeometry(
        window_width=config.window_width_m,
        radiator_size=config.radiator_size_m,
        radiator_offset_r=r_m,
        radiator_center_x=config.radiator_center_x_m,
    )
    masks = build_room_masks(grid, geom)
    u_end_C = u_end_K - 273.15

    y = np.linspace(0.0, config.room_height_m, u_end_C.shape[0])
    x = np.linspace(0.0, config.room_width_m, u_end_C.shape[1])

    overlays = {
        "boundary": _mask_points(masks.boundary, x, y),
        "window": _mask_points(masks.windows, x, y),
        "radiator": _mask_points(masks.radiator, x, y),
    }
    return MapInputs(u_end_C=u_end_C, x=x, y=y, overlays=overlays)
