# Ten plik laduje stale fizyczne z data/physical_constants.csv.
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class PhysicalConstants:
    """Container for physical constants used in the project."""

    specific_gas_constant_air: float
    specific_heat_air_cp: float
    atmospheric_pressure: float
    thermal_diffusivity_air: float


def _constants_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "physical_constants.csv"


def load_physical_constants(
    path: Optional[Path] = None,
) -> PhysicalConstants:
    """Load physical constants from the CSV file."""
    csv_path = path or _constants_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing constants file: {csv_path}")

    values: Dict[str, float] = {}
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("name")
            value = row.get("value")
            if name and value:
                values[name] = float(value)

    return PhysicalConstants(
        specific_gas_constant_air=values["specific_gas_constant_air"],
        specific_heat_air_cp=values["specific_heat_air_cp"],
        atmospheric_pressure=values["atmospheric_pressure"],
        thermal_diffusivity_air=values["thermal_diffusivity_air"],
    )


def load_physical_constants_safe(
    path: Optional[Path] = None,
) -> PhysicalConstants:
    """Load constants with a safe fallback if the file is missing."""
    try:
        return load_physical_constants(path)
    except FileNotFoundError:
        return PhysicalConstants(
            specific_gas_constant_air=287.0,
            specific_heat_air_cp=1005.0,
            atmospheric_pressure=101325.0,
            thermal_diffusivity_air=2.1e-5,
        )
