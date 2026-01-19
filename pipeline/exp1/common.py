# Ten plik zawiera wspolne funkcje i konfiguracje dla eksp1.
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from pipeline.solver import Grid


@dataclass(frozen=True)
class Exp1VizConfig:
    """Configuration container for exp1 visualization and simulation inputs."""
    room_width_m: float
    room_height_m: float
    hx: float
    t_end: float
    dt: float
    alpha: float
    u_out_C: float
    u0_C: float
    setpoint_C: float
    radiator_coeff: float
    window_width_m: float
    radiator_size_m: tuple[float, float]
    radiator_center_x_m: Optional[float]
    bc_kind: str


def grid_from_room(width_m: float, height_m: float, hx: float) -> Grid:
    """Create a numerical grid from room dimensions and spatial step."""
    nx = int(round(width_m / hx)) + 1
    ny = int(round(height_m / hx)) + 1
    return Grid(nx=nx, ny=ny, hx=hx)


def load_results(csv_path: Path) -> List[dict]:
    """Load exp1 results from a CSV file."""
    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {csv_path}.")
    return rows


def save_table_md(rows: Iterable[dict], output_path: Path) -> None:
    """Write a Markdown summary table for exp1 results."""
    rows = list(rows)
    lines = ["| r_m | mu_C | sigma_C |", "| --- | --- | --- |"]
    for row in rows:
        r = float(row["r_m"])
        mu = float(row["mu_C"])
        sigma = float(row["sigma_C"])
        lines.append(f"| {r:.3f} | {mu:.3f} | {sigma:.3f} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def select_map_r_values(rows: Sequence[dict], raw_values: Optional[str]) -> List[float]:
    """Select r values for temperature maps based on input or defaults."""
    if raw_values:
        values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
        if not values:
            raise ValueError("Empty --map-r-values list.")
        return values
    r_vals = sorted(float(row["r_m"]) for row in rows)
    if len(r_vals) <= 2:
        return r_vals
    return [r_vals[0], r_vals[len(r_vals) // 2], r_vals[-1]]


def C_to_K(temp_c: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + 273.15
