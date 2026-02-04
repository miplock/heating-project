from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pipeline").exists():
            return parent
    return current.parent


def _is_metadata_record(record: object) -> bool:
    return (
        isinstance(record, dict)
        and record.get("type") == "metadata"
        and "simulation_parameters" in record
    )


def _iter_pkl_records(pkl_path: Path):
    with pkl_path.open("rb") as pkl_file:
        try:
            while True:
                yield pickle.load(pkl_file)
        except EOFError:
            return


def _load_metadata(pkl_path: Path) -> Optional[dict]:
    for record in _iter_pkl_records(pkl_path):
        if _is_metadata_record(record):
            return record
    return None


def _extract_steps(pkl_path: Path) -> list[dict]:
    steps: list[dict] = []
    for record in _iter_pkl_records(pkl_path):
        if _is_metadata_record(record):
            continue
        steps.append(record)
    if not steps:
        raise RuntimeError(f"No steps found in {pkl_path}")
    return steps


def create_sd_temperature_figure(
    pkl_path: Path,
    params_csv_path: Optional[Path] = None,
):
    from pipeline.exp1.runner.simulation_parameters import load_simulation_parameters

    metadata = _load_metadata(pkl_path)
    if metadata is not None:
        params = metadata["simulation_parameters"]
    else:
        if params_csv_path is None:
            raise RuntimeError("params_csv_path is required when PKL has no metadata")
        params = load_simulation_parameters(params_csv_path)

    time_step = float(params["time_step"])
    steps = _extract_steps(pkl_path)

    times = []
    sd_celsius = []
    for step in steps:
        u_n = np.asarray(step["u_n"], dtype=np.float32)
        sd_k = float(np.std(u_n))
        sd_c = sd_k
        times.append(float(step["step"]) * time_step)
        sd_celsius.append(sd_c)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=times,
                y=sd_celsius,
                mode="lines+markers",
                name="Temperature standard deviation",
                line=dict(color="royalblue", width=2),
                marker=dict(size=5),
            )
        ]
    )
    fig.update_layout(
        title="Temperature standard deviation over time",
        xaxis_title="Time (s)",
        yaxis_title="Temperature standard deviation (C)",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig
