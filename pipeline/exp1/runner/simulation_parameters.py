from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


def load_simulation_parameters(csv_path: Path) -> Dict[str, float]:
    """Load simulation parameters from a CSV file into a name -> value mapping."""
    parameters: Dict[str, float] = {}
    int_fields = {"time_steps"}
    bool_fields = {"use_iterative_solver"}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            name = row["name"].strip()
            value_raw = row["value"]
            if name in int_fields:
                value = int(float(value_raw))
            elif name in bool_fields:
                normalized = value_raw.strip().lower()
                if normalized in {"true", "1", "yes", "y"}:
                    value = True
                elif normalized in {"false", "0", "no", "n"}:
                    value = False
                else:
                    raise ValueError(
                        f"Invalid boolean value for {name}: {value_raw!r}"
                    )
            else:
                value = float(value_raw)
            parameters[name] = value
    return parameters
