"""Output helpers for Experiment 1."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def write_csv(rows: Iterable[dict], output_path: Path) -> None:
    """Write experiment summary rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write.")
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_maps(
    *,
    output_path: Path,
    r_values: Iterable[float],
    t_values: np.ndarray,
    u_time_by_r: dict[float, np.ndarray],
) -> None:
    """Persist full temperature maps to a .npz file."""
    r_list = [float(r) for r in r_values]
    u_stack = np.stack([u_time_by_r[float(r)] for r in r_values])
    np.savez(
        output_path,
        r_values=np.array(r_list),
        t_values=t_values,
        u_time_K=u_stack,
    )
