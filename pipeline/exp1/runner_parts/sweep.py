"""Run the Experiment 1 parameter sweep."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from pipeline.exp1.progress import ProgressTracker
from pipeline.geometry import RoomGeometry, build_room_masks
from pipeline.metrics import mean_temp_C, std_temp_C
from pipeline.solver import (
    BoundaryParams,
    HeatEquationSolver2D,
    PhysicalParams,
    RadiatorParams,
    TimeGrid,
)

from .config import Exp1Config
from .utils import C_to_K, default_dt, grid_from_room


def run_exp1(
    config: Exp1Config,
    r_values: Sequence[float],
    *,
    progress: bool = False,
    collect_maps: bool = False,
) -> tuple[List[dict], dict[float, np.ndarray] | None, np.ndarray | None]:
    """Run simulations for each radiator offset and collect statistics."""
    # Ustal siatke, czas i parametry fizyczne wspolne dla wszystkich prob.
    grid = grid_from_room(config.room_width_m, config.room_height_m, config.hx)
    # Wylicz dt albo skorzystaj z wartosci domyslnej.
    dt = config.dt if config.dt is not None else default_dt()
    # Zbuduj siatke czasu dla calej symulacji.
    time = TimeGrid(dt=dt, t_end=config.t_end)
    # Parametr dyfuzji ciepla w modelu.
    phys = PhysicalParams(alpha=config.alpha)

    # Konfiguracja warunkow brzegowych z temperatury na zewnatrz.
    if config.bc_kind == "dirichlet":
        # Dla Dirichleta zadana jest temperatura na brzegach.
        bc = BoundaryParams(
            kind="dirichlet",
            dirichlet_value=C_to_K(config.u_out_C),
        )
    else:
        # Inne typy BC nie potrzebuja wartosci temperatury.
        bc = BoundaryParams(kind=config.bc_kind)

    # Startowe pole temperatury.
    u0 = np.full(grid.shape, C_to_K(config.u0_C))

    # Lista wynikow do CSV.
    results = []
    # Mapa czasu tylko gdy zbieramy pelne przebiegi.
    u_time_by_r: dict[float, np.ndarray] | None = None
    # Wektor czasow z symulacji (wypelniany raz).
    t_values: np.ndarray | None = None
    if collect_maps:
        # Przygotuj slownik na przebiegi dla kazdego r.
        u_time_by_r = {}
    # Calkowita liczba prob.
    total = len(r_values)
    # Opcjonalny pasek postepu.
    tracker = ProgressTracker("exp1", total) if progress else None
    # Solver tworzymy raz i tylko podmieniamy maski.
    solver: HeatEquationSolver2D | None = None
    for idx, r in enumerate(r_values, start=1):
        if tracker is not None:
            # Zaktualizuj postep i wyswietl biezace r.
            suffix = f"r={r:.3f}m"
            tracker.update(idx, suffix=suffix)
        # Geometria pokoju i maski zalezne od przesuniecia grzejnika.
        geom = RoomGeometry(
            window_width=config.window_width_m,
            radiator_size=config.radiator_size_m,
            radiator_offset_r=r,
            radiator_center_x=config.radiator_center_x_m,
        )
        # Zbuduj maski domeny, okna i grzejnika.
        masks = build_room_masks(grid, geom)
        # Parametry grzejnika (moc i zadana temperatura).
        radiator = RadiatorParams(
            enabled=True,
            setpoint=C_to_K(config.setpoint_C),
            coeff=config.radiator_coeff,
        )

        # Pierwsza iteracja tworzy solver, kolejne tylko podmieniaja maski.
        if solver is None:
            # Inicjalizacja solvera dla wspolnych parametrow.
            solver = HeatEquationSolver2D(
                grid=grid,
                time=time,
                phys=phys,
                bc=bc,
                masks=masks,
                radiator=radiator,
            )
        else:
            # Aktualizacja masek i parametrow grzejnika.
            solver.masks = masks
            solver.radiator = radiator
        # Symulacja: albo pelne mapy w czasie, albo tylko koncowy stan.
        if collect_maps:
            # Zapisuj kazdy krok czasu.
            out = solver.simulate(u0, store_every=1)
        else:
            # Zapisz tylko ostatni stan.
            out = solver.simulate(u0, store_every=time.n_steps)
        # Ostatnie pole temperatury.
        u_end = out["U"][-1]

        # Statystyki temperatury w domenie pokoju.
        # Srednia temperatura.
        mu_T = mean_temp_C(u_end, mask=masks.domain)
        # Odchylenie standardowe temperatury.
        sigma_T = std_temp_C(u_end, mask=masks.domain)
        if u_time_by_r is not None:
            # Przechowaj cale przebiegi w czasie dla danego r.
            u_time_by_r[float(r)] = np.stack(out["U"])
            if t_values is None:
                # Czasy zapisujemy tylko raz.
                t_values = np.array(out["t"], dtype=float)

        # Dodaj wiersz wynikow do listy.
        results.append(
            {
                "r_m": float(r),
                "mu_C": float(mu_T),
                "sigma_C": float(sigma_T),
            }
        )

    if tracker is not None:
        # Zakonczenie paska postepu.
        tracker.finish()
    return results, u_time_by_r, t_values
