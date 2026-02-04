from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from pipeline.shared.plots.plot_sd_temperature import (
    _project_root,
    create_sd_temperature_figure,
)


def save_sd_temperature_figure(
    pkl_path: Path | None = None,
    params_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    project_root = _project_root()
    sys.path.insert(0, str(project_root))

    if pkl_path is None:
        pkl_path = project_root / "data/shared/processed/heat_results.pkl"
    if params_path is None:
        params_path = project_root / "data/Q1/raw/sim1_params_def.csv"
    if output_path is None:
        output_path = project_root / "data/Q1/figures/sd_temperature.pkl"

    fig = create_sd_temperature_figure(pkl_path, params_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(fig, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved temperature SD figure to {output_path}")
    return output_path


def main() -> int:
    project_root = _project_root()
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="Create a Plotly temperature SD chart and save it as a PKL file"
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=project_root / "data/shared/processed/heat_results.pkl",
        help="Path to heat_results.pkl",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=project_root / "data/Q1/raw/sim1_params_def.csv",
        help="Path to simulation parameters CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/Q1/figures/sd_temperature.pkl",
        help="Output PKL path",
    )
    args = parser.parse_args()

    save_sd_temperature_figure(args.pkl, args.params, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
