from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from pipeline.exp1.plots.plot_heatmap import (
    _project_root,
    create_heatmap_figure,
)


def main() -> int:
    project_root = _project_root()
    sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="Create a Plotly heatmap and save it as a PKL file"
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
        "--step",
        type=int,
        default=None,
        help="Step index to plot (omit for slider, -1 means last step)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to sample uniformly across time",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=600,
        help="Max grid dimension for plotting (set 0 to disable downsampling)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/Q1/figures/heatmap_last.pkl",
        help="Output PKL path",
    )
    args = parser.parse_args()

    fig = create_heatmap_figure(
        args.pkl,
        args.params,
        step_index=args.step,
        frame_count=args.frames,
        max_plot_dim=args.max_dim,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(fig, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved heatmap figure to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
