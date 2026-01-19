# Ten plik przygotowuje dane do wykresow metryk exp1.
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def build_metric_series(rows: Iterable[dict], field: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build x/y series for a metric plot from exp1 result rows."""
    rows = list(rows)
    r_vals = np.array([float(row["r_m"]) for row in rows])
    y_vals = np.array([float(row[field]) for row in rows])
    return r_vals, y_vals
