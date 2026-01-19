# Ten plik spina generowanie wykresow i tabel dla exp1.
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from pipeline.exp1.common import (
    Exp1VizConfig,
    load_results,
    save_table_md,
    select_map_r_values,
)
from pipeline.exp1.experimental_params import resolve_params
from pipeline.exp1.plots.maps import generate_map_plots_from_time
from pipeline.exp1.plots.metrics import generate_metric_plots


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for exp1 visualization."""
    params = resolve_params()
    parser = argparse.ArgumentParser(
        description="Generate tables, plots, and temperature maps for exp1."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/exp1_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp1_figures"),
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=Path("data/exp1_table.md"),
    )
    parser.add_argument("--no-mu-plot", action="store_true")
    parser.add_argument("--map-r-values", type=str, default=None)
    parser.add_argument("--use-experimental", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--maps-input",
        type=Path,
        default=Path("data/exp1_maps.npz"),
    )

    parser.add_argument("--room-width", type=float, default=params.room_width)
    parser.add_argument("--room-height", type=float, default=params.room_height)
    parser.add_argument("--hx", type=float, default=params.hx)
    parser.add_argument("--t-end", type=float, default=params.t_end)
    parser.add_argument("--dt", type=float, default=params.dt)
    parser.add_argument("--alpha", type=float, default=params.alpha)
    parser.add_argument("--u-out-c", type=float, default=params.u_out_c)
    parser.add_argument("--u0-c", type=float, default=params.u0_c)
    parser.add_argument(
        "--setpoint-c",
        type=float,
        default=params.setpoint_c,
    )
    parser.add_argument("--rad-coeff", type=float, default=params.rad_coeff)
    parser.add_argument(
        "--window-width",
        type=float,
        default=params.window_width,
    )
    parser.add_argument("--rad-width", type=float, default=params.rad_width)
    parser.add_argument("--rad-height", type=float, default=params.rad_height)
    parser.add_argument(
        "--rad-center-x",
        type=float,
        default=params.rad_center_x,
    )
    parser.add_argument("--bc-kind", type=str, default=params.bc_kind)
    return parser.parse_args()


def _load_map_data(
    maps_path: Path,
    map_r_values: list[float],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    # Wczytanie map z pliku .npz i mapowanie po wartosciach r.
    if not maps_path.exists():
        raise FileNotFoundError(
            "Missing maps file. Run exp1 runner with maps enabled."
        )
    data = np.load(maps_path)
    r_values = data["r_values"]
    t_values = data["t_values"]
    u_time = data["u_time_K"]
    if u_time.shape[0] != r_values.shape[0]:
        raise ValueError("Maps file has inconsistent shapes.")
    if u_time.shape[1] != t_values.shape[0]:
        raise ValueError("Maps file has inconsistent time axis.")

    u_time_by_r: dict[float, np.ndarray] = {}
    for r_m in map_r_values:
        idxs = np.where(np.isclose(r_values, r_m))[0]
        if idxs.size == 0:
            raise ValueError(f"Missing map for r={r_m:.3f}.")
        u_time_by_r[float(r_m)] = u_time[int(idxs[0])]
    return t_values, u_time_by_r


def main() -> None:
    """Run the full exp1 visualization pipeline."""
    args = parse_args()
    preset = resolve_params(use_experimental=args.use_experimental)
    # Wczytanie wynikow i zapis tabeli.
    rows = load_results(args.input)
    save_table_md(rows, args.table_out)

    # Wykresy metryk (sigma i opcjonalnie mu).
    generate_metric_plots(rows, args.output_dir, include_mu=not args.no_mu_plot)

    config = Exp1VizConfig(
        room_width_m=preset.room_width,
        room_height_m=preset.room_height,
        hx=preset.hx,
        t_end=preset.t_end,
        dt=preset.dt,
        alpha=preset.alpha,
        u_out_C=preset.u_out_c,
        u0_C=preset.u0_c,
        setpoint_C=preset.setpoint_c,
        radiator_coeff=preset.rad_coeff,
        window_width_m=preset.window_width,
        radiator_size_m=(preset.rad_width, preset.rad_height),
        radiator_center_x_m=preset.rad_center_x,
        bc_kind=preset.bc_kind,
    )
    # Mapy temperatury dla wybranych wartosci r.
    map_r_values = select_map_r_values(rows, args.map_r_values)
    t_values, u_time_by_r = _load_map_data(args.maps_input, map_r_values)
    generate_map_plots_from_time(
        config=config,
        output_dir=args.output_dir,
        t_values=t_values,
        u_time_by_r=u_time_by_r,
        progress=args.progress,
    )

    print(
        f"Wrote Plotly pickle files to {args.output_dir} "
        f"and table to {args.table_out}"
    )


if __name__ == "__main__":
    main()
