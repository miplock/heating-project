from __future__ import annotations

import csv
from pathlib import Path


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pipeline").exists():
            return parent
    return current.parent


PHYSICAL_CONSTANTS_CSV_PATH = _project_root() / "data/shared/raw/physical_constants.csv"


def write_physical_constants_csv(csv_path: Path) -> None:
    rows = [
        {
            "name": "specific_gas_constant_air",
            "symbol": "r",
            "value": "287",
            "units": "J/(kg*K)",
            "source": "https://www.engineeringtoolbox.com/air-properties-d_156.html",
        },
        {
            "name": "specific_heat_air_cp",
            "symbol": "c",
            "value": "1005",
            "units": "J/(kg*K)",
            "source": "https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html",
        },
        {
            "name": "atmospheric_pressure",
            "symbol": "p",
            "value": "101325", 
            "units": "Pa",
            "source": "https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html",
        },
        {
            "name": "thermal_diffusivity_air",
            "symbol": "alpha",
            "value": "2.1e-5",
            "units": "m^2/s",
            "source": "https://www.engineeringtoolbox.com/air-thermal-diffusivity-d_201.html",
        },
        {
            "name": "thermal_conductivity_air",
            "symbol": "lambda",
            "value": "0.024",
            "units": "W/(m*K)",
            "source": "https://www.engineeringtoolbox.com/air-thermal-conductivity-d_429.html",
        },
        {
            "name": "heat_transfer_coefficient_concrete",
            "symbol": "h",
            "value": "0.0024",
            "units": "W/(m^2*K)",
            "source": "",
        },
        {
            "name": "heat_transfer_coefficient_glass",
            "symbol": "h",
            "value": "6.5",
            "units": "W/(m^2*K)",
            "source": "",
        },
        {
            "name": "radiator_power",
            "symbol": "P",
            "value": "2000",
            "units": "W",
            "source": "",
        },
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["name", "symbol", "value", "units", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Physical constants CSV saved to {csv_path}")


if __name__ == "__main__":
    write_physical_constants_csv(PHYSICAL_CONSTANTS_CSV_PATH)
