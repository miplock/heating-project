# Ten plik generuje wykresy metryk exp1 i zapisuje je do plikow .pkl.
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go

from pipeline.exp1.metrics_logic import build_metric_series
from pipeline.plots.theme import apply_plotly_theme


def plot_metric(
    rows: Iterable[dict],
    *,
    field: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> None:
    """Create and persist a single metric plot as a Plotly pickle."""
    r_vals, y_vals = build_metric_series(rows, field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=r_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(width=2),
            )
        ]
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="r [m]",
        yaxis_title=ylabel,
    )
    with output_path.open("wb") as handle:
        pickle.dump(fig, handle)


def generate_metric_plots(rows: Iterable[dict], output_dir: Path, include_mu: bool) -> None:
    """Generate sigma(T) and optionally mu(T) plots for exp1."""
    sigma_path = output_dir / "exp1_sigma_vs_r.pkl"
    plot_metric(
        rows,
        field="sigma_C",
        ylabel="sigma(T) [°C]",
        output_path=sigma_path,
        title="exp1: sigma(T) vs r",
    )

    if include_mu:
        mu_path = output_dir / "exp1_mu_vs_r.pkl"
        plot_metric(
            rows,
            field="mu_C",
            ylabel="mu(T) [°C]",
            output_path=mu_path,
            title="exp1: mu(T) vs r",
        )
