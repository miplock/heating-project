"""CLI parsing for Experiment 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.exp1.experimental_params import resolve_params


def parse_args() -> argparse.Namespace:
    params = resolve_params()
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1: sweep radiator offset r and save mu(T), sigma(T)."
        )
    )
    parser.add_argument("--room-width", type=float, default=params.room_width)
    parser.add_argument("--room-height", type=float, default=params.room_height)
    parser.add_argument("--hx", type=float, default=params.hx)
    parser.add_argument("--t-end", type=float, default=params.t_end)
    parser.add_argument("--dt", type=float, default=params.dt)
    parser.add_argument("--alpha", type=float, default=params.alpha)
    parser.add_argument("--u-out-c", type=float, default=params.u_out_c)
    parser.add_argument("--u0-c", type=float, default=params.u0_c)
    parser.add_argument("--setpoint-c", type=float, default=params.setpoint_c)
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
    parser.add_argument("--r-values", type=str, default=None)
    parser.add_argument("--r-min", type=float, default=None)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--r-count", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exp1_results.csv"),
    )
    parser.add_argument("--use-experimental", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--maps-out",
        type=Path,
        default=Path("data/exp1_maps.npz"),
    )
    parser.add_argument("--no-maps", action="store_true")
    return parser.parse_args()
