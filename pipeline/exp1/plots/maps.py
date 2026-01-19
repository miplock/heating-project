# Ten plik generuje mapy temperatury exp1 i zapisuje je do plikow .pkl.
from __future__ import annotations

import pickle
from pathlib import Path

import plotly.graph_objects as go

from pipeline.exp1.common import Exp1VizConfig
from pipeline.exp1.maps_math import (
    build_map_inputs,
    build_map_inputs_from_u_end,
)
from pipeline.exp1.progress import ProgressTracker
from pipeline.plots.theme import apply_plotly_theme


def generate_map_plots(
    *,
    config: Exp1VizConfig,
    output_dir: Path,
    map_r_values: list[float],
    progress: bool = False,
) -> None:
    """Generate and persist temperature maps for selected r values."""
    # Dla kazdego r tworzymy mape i zapisujemy ja do osobnego pliku.
    total = len(map_r_values)
    tracker = ProgressTracker("maps", total) if progress else None
    for idx, r_m in enumerate(map_r_values, start=1):
        if tracker is not None:
            suffix = f"r={r_m:.2f}m"
            tracker.update(idx, suffix=suffix)
        # Skrot nazwy pliku, zeby latwo widziec r w nazwie.
        out_name = f"exp1_map_r_{r_m:.2f}m.pkl"
        # Pelna sciezka do pliku wyjsciowego.
        out_path = output_dir / out_name
        # Pobranie temperatur i obrysow masek z czesci obliczen.
        map_inputs = build_map_inputs(config, r_m)
        # Zapewnienie istnienia katalogu wyjsciowego.
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = _build_map_figure(map_inputs, r_m)
        # Zapis obiektu Plotly do pliku .pkl.
        with out_path.open("wb") as handle:
            pickle.dump(fig, handle)
    if tracker is not None:
        tracker.finish()


def generate_map_plots_from_data(
    *,
    config: Exp1VizConfig,
    output_dir: Path,
    u_end_by_r: dict[float, np.ndarray],
    progress: bool = False,
) -> None:
    """Generate maps using precomputed temperature fields."""
    items = list(u_end_by_r.items())
    total = len(items)
    tracker = ProgressTracker("maps", total) if progress else None
    for idx, (r_m, u_end_K) in enumerate(items, start=1):
        if tracker is not None:
            suffix = f"r={r_m:.2f}m"
            tracker.update(idx, suffix=suffix)
        out_name = f"exp1_map_r_{r_m:.2f}m.pkl"
        out_path = output_dir / out_name
        map_inputs = build_map_inputs_from_u_end(config, r_m, u_end_K)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = _build_map_figure(map_inputs, r_m)
        with out_path.open("wb") as handle:
            pickle.dump(fig, handle)
    if tracker is not None:
        tracker.finish()


def generate_map_plots_from_time(
    *,
    config: Exp1VizConfig,
    output_dir: Path,
    t_values: np.ndarray,
    u_time_by_r: dict[float, np.ndarray],
    progress: bool = False,
) -> None:
    """Generate animated maps with a time slider."""
    items = list(u_time_by_r.items())
    total = len(items)
    tracker = ProgressTracker("maps", total) if progress else None
    for idx, (r_m, u_time_K) in enumerate(items, start=1):
        if tracker is not None:
            suffix = f"r={r_m:.2f}m"
            tracker.update(idx, suffix=suffix)
        out_name = f"exp1_map_r_{r_m:.2f}m.pkl"
        out_path = output_dir / out_name
        map_inputs = build_map_inputs_from_u_end(config, r_m, u_time_K[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        u_time_C = u_time_K - 273.15
        fig = _build_map_animation(map_inputs, r_m, t_values, u_time_C)
        with out_path.open("wb") as handle:
            pickle.dump(fig, handle)
    if tracker is not None:
        tracker.finish()


def _build_map_figure(map_inputs, r_m: float) -> go.Figure:
    # Buduje figure Plotly z przygotowanych danych mapy.
    hover = (
        "x=%{x:.2f} m<br>"
        "y=%{y:.2f} m<br>"
        "T=%{z:.2f} °C<extra></extra>"
    )
    heatmap = go.Heatmap(
        z=map_inputs.u_end_C,
        x=map_inputs.x,
        y=map_inputs.y,
        colorscale="Inferno",
        colorbar=dict(title="T [°C]"),
        hovertemplate=hover,
    )
    fig = go.Figure(data=[heatmap])
    color_map = {
        "boundary": "white",
        "window": "cyan",
        "radiator": "lime",
    }
    for name, (xs, ys) in map_inputs.overlays.items():
        if xs.size == 0:
            continue
        if name == "radiator":
            x0 = xs.min()
            x1 = xs.max()
            y0 = ys.min()
            y1 = ys.max()
            rect_x = [x0, x1, x1, x0, x0]
            rect_y = [y0, y0, y1, y1, y0]
            fig.add_trace(
                go.Scatter(
                    x=rect_x,
                    y=rect_y,
                    mode="lines",
                    line=dict(color=color_map.get(name, "lime"), width=2),
                    name=name,
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color=color_map.get(name, "white"), size=3),
                name=name,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    apply_plotly_theme(fig)
    fig.update_layout(
        title=f"Mapa temperatury T dla r = {r_m:.2f} m",
        title_x=0.5,
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis_autorange="reversed",
    )
    return fig


def _build_map_animation(
    map_inputs,
    r_m: float,
    t_values: np.ndarray,
    u_time_C: np.ndarray,
) -> go.Figure:
    # Buduje figure Plotly z animacja czasowa.
    hover = (
        "x=%{x:.2f} m<br>"
        "y=%{y:.2f} m<br>"
        "T=%{z:.2f} °C<extra></extra>"
    )
    heatmap = go.Heatmap(
        z=u_time_C[0],
        x=map_inputs.x,
        y=map_inputs.y,
        colorscale="Inferno",
        colorbar=dict(title="T [°C]"),
        hovertemplate=hover,
    )
    fig = go.Figure(data=[heatmap])
    _add_overlays(fig, map_inputs)

    frames = []
    for idx, t in enumerate(t_values):
        frame = go.Frame(
            data=[
                go.Heatmap(
                    z=u_time_C[idx],
                    x=map_inputs.x,
                    y=map_inputs.y,
                    colorscale="Inferno",
                    hovertemplate=hover,
                )
            ],
            name=str(idx),
            traces=[0],
        )
        frames.append(frame)

    steps = []
    for idx, t in enumerate(t_values):
        steps.append(
            {
                "method": "animate",
                "args": [
                    [str(idx)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{t:.2f}s",
            }
        )

    fig.frames = frames
    apply_plotly_theme(fig)
    fig.update_layout(
        title=f"Mapa temperatury T dla r = {r_m:.2f} m",
        title_x=0.5,
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        yaxis_autorange="reversed",
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "x": 0.1,
                "y": 0.0,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": 0.05,
                "xanchor": "left",
                "yanchor": "bottom",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def _add_overlays(fig: go.Figure, map_inputs) -> None:
    # Dodaje maski geometrii jako osobne slady.
    color_map = {
        "boundary": "white",
        "window": "cyan",
        "radiator": "lime",
    }
    for name, (xs, ys) in map_inputs.overlays.items():
        if xs.size == 0:
            continue
        if name == "radiator":
            x0 = xs.min()
            x1 = xs.max()
            y0 = ys.min()
            y1 = ys.max()
            rect_x = [x0, x1, x1, x0, x0]
            rect_y = [y0, y0, y1, y1, y0]
            fig.add_trace(
                go.Scatter(
                    x=rect_x,
                    y=rect_y,
                    mode="lines",
                    line=dict(color=color_map.get(name, "lime"), width=2),
                    name=name,
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color=color_map.get(name, "white"), size=3),
                name=name,
                hoverinfo="skip",
                showlegend=True,
            )
        )
