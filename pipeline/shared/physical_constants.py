from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


def load_physical_constants(csv_path: Path) -> Dict[str, float]:
    """Load physical constants from a CSV file into a name -> value mapping."""
    constants: Dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            name = row["name"].strip()
            value = float(row["value"])
            constants[name] = value
    return constants
