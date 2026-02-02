from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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


def _load_metadata_and_step_count(pkl_path: Path) -> tuple[Optional[dict], int]:
    metadata = None
    step_count = 0
    for record in _iter_pkl_records(pkl_path):
        if _is_metadata_record(record):
            metadata = record
        else:
            step_count += 1
    return metadata, step_count


def _format_elapsed_time(total_seconds: float) -> str:
    seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}min {secs:02d}s"


def _select_step_indices(step_count: int, frame_count: int) -> Optional[set[int]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count >= step_count:
        return None
    idx = np.linspace(0, step_count - 1, num=frame_count)
    idx = np.unique(np.round(idx).astype(int))
    return set(int(i) for i in idx)


def _downsample_grid(
    grid: np.ndarray, max_plot_dim: Optional[int]
) -> tuple[np.ndarray, int]:
    if max_plot_dim is None or max_plot_dim <= 0:
        return grid, 1
    max_dim = max(grid.shape)
    if max_dim <= max_plot_dim:
        return grid, 1
    stride = int(np.ceil(max_dim / max_plot_dim))
    return grid[::stride, ::stride], stride


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _format_tick(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def _axis_ticks(length_m: float, space_step: float, stride: int) -> tuple[list[float], list[str]]:
    if length_m <= 0 or space_step <= 0:
        return [], []
    target_ticks = 6
    approx_step = length_m / target_ticks
    mult = max(1, int(round(approx_step / space_step)))
    tick_step_m = mult * space_step
    tick_m = np.arange(0.0, length_m + 1e-9, tick_step_m)
    tick_vals = (tick_m / space_step) / stride
    tick_text = [_format_tick(val) for val in tick_m]
    return tick_vals.tolist(), tick_text


def _build_border_shapes(
    grid_shape: tuple[int, int],
    *,
    border_thickness: float,
    window_segment: Optional[tuple[float, float]] = None,
    radiator_rect: Optional[tuple[float, float, float, float]] = None,
):
    ny, nx = grid_shape
    x0 = 0.0
    x1 = float(nx - 1)
    y0 = 0.0
    y1 = float(ny - 1)
    t = float(border_thickness)

    def rect(x0v: float, x1v: float, y0v: float, y1v: float, fill: str):
        return dict(
            type="rect",
            xref="x",
            yref="y",
            x0=x0v,
            x1=x1v,
            y0=y0v,
            y1=y1v,
            fillcolor=fill,
            line=dict(color="black", width=1),
            layer="above",
        )

    shapes = [
        rect(x0 - t, x1 + t, y0 - t, y0, "white"),
        rect(x0 - t, x0, y0, y1, "white"),
        rect(x1, x1 + t, y0, y1, "white"),
    ]

    if window_segment is None:
        shapes.append(rect(x0 - t, x1 + t, y1, y1 + t, "white"))
    else:
        wx0, wx1 = window_segment
        wx0 = _clamp(wx0, x0, x1)
        wx1 = _clamp(wx1, x0, x1)
        if wx1 <= wx0:
            shapes.append(rect(x0 - t, x1 + t, y1, y1 + t, "white"))
        else:
            shapes.append(rect(x0 - t, wx0, y1, y1 + t, "white"))
            shapes.append(rect(wx1, x1 + t, y1, y1 + t, "white"))
            shapes.append(rect(wx0, wx1, y1, y1 + t, "royalblue"))

    if radiator_rect is not None:
        rx0, rx1, ry0, ry1 = radiator_rect
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=rx0,
                x1=rx1,
                y0=ry0,
                y1=ry1,
                fillcolor="red",
                line=dict(color="black", width=1),
                layer="above",
            )
        )
    return shapes


def create_heatmap_figure(
    pkl_path: Path,
    params_csv_path: Optional[Path],
    *,
    step_index: int | None = None,
    frame_count: int = 20,
    max_plot_dim: Optional[int] = 600,
):
    from pipeline.exp1.runner.simulation_parameters import load_simulation_parameters

    metadata = None
    steps: list[dict] = []
    steps_idx: list[int] = []

    if step_index is None:
        metadata, step_count = _load_metadata_and_step_count(pkl_path)
        if step_count == 0:
            raise RuntimeError(f"No steps found in {pkl_path}")
        selected_indices = _select_step_indices(step_count, frame_count)
        step_counter = 0
        for record in _iter_pkl_records(pkl_path):
            if _is_metadata_record(record):
                continue
            if selected_indices is None or step_counter in selected_indices:
                steps.append(record)
                steps_idx.append(record["step"])
            step_counter += 1
    else:
        last_step = None
        for record in _iter_pkl_records(pkl_path):
            if _is_metadata_record(record):
                metadata = record
                continue
            if step_index >= 0:
                if record.get("step") == step_index:
                    steps = [record]
                    steps_idx = [record["step"]]
                    break
            else:
                last_step = record
        if step_index >= 0 and not steps:
            raise RuntimeError(f"Step {step_index} not found in {pkl_path}")
        if step_index < 0:
            if last_step is None:
                raise RuntimeError(f"No steps found in {pkl_path}")
            steps = [last_step]
            steps_idx = [last_step["step"]]

    if metadata is not None:
        params = metadata["simulation_parameters"]
    else:
        if params_csv_path is None:
            raise RuntimeError("params_csv_path is required when PKL has no metadata")
        params = load_simulation_parameters(params_csv_path)
    nx = int(params["room_width"] / params["space_step"]) + 1
    ny = int(params["room_length"] / params["space_step"]) + 1
    time_step = float(params["time_step"])

    grids = []
    vmin = None
    vmax = None
    stride_used = None
    radiator_rect = None

    for step in steps:
        u_n = np.asarray(step["u_n"], dtype=np.float32) - np.float32(273.15)
        if u_n.size != nx * ny:
            raise RuntimeError(
                f"Unexpected vector size {u_n.size}, expected {nx * ny}"
            )
        grid = u_n.reshape(ny, nx)
        grid, stride = _downsample_grid(grid, max_plot_dim)
        if stride_used is None:
            stride_used = stride
        grids.append(grid)
        grid_for_scale = grid
        if "radiator_pos_x" in params and "radiator_pos_y" in params:
            rx0 = int(round(float(params["radiator_pos_x"]) / params["space_step"]))
            rx1 = int(
                round(
                    (float(params["radiator_pos_x"]) + float(params["radiator_size_x"]))
                    / params["space_step"]
                )
            )
            ry0 = int(round(float(params["radiator_pos_y"]) / params["space_step"]))
            ry1 = int(
                round(
                    (float(params["radiator_pos_y"]) + float(params["radiator_size_y"]))
                    / params["space_step"]
                )
            )
            if stride_used is None:
                stride_used = 1
            rx0_ds = int(np.floor(rx0 / stride_used))
            rx1_ds = int(np.ceil(rx1 / stride_used))
            ry0_ds = int(np.floor(ry0 / stride_used))
            ry1_ds = int(np.ceil(ry1 / stride_used))
            rx0_ds = max(0, min(rx0_ds, grid.shape[1] - 1))
            rx1_ds = max(0, min(rx1_ds, grid.shape[1] - 1))
            ry0_ds = max(0, min(ry0_ds, grid.shape[0] - 1))
            ry1_ds = max(0, min(ry1_ds, grid.shape[0] - 1))
            if radiator_rect is None:
                radiator_rect = (rx0_ds, rx1_ds, ry0_ds, ry1_ds)
            grid_for_scale = grid.copy()
            grid_for_scale[ry0_ds : ry1_ds + 1, rx0_ds : rx1_ds + 1] = np.nan
        current_min = float(np.nanmin(grid_for_scale))
        current_max = float(np.nanmax(grid_for_scale))
        vmin = current_min if vmin is None else min(vmin, current_min)
        vmax = current_max if vmax is None else max(vmax, current_max)

    grid_shape = grids[0].shape
    border_thickness = max(1.0, round(min(grid_shape) * 0.03, 2))
    window_segment = None
    if stride_used is None:
        stride_used = 1
    if "window_pos_x" in params and "window_width" in params:
        window_x0_idx = int(round(float(params["window_pos_x"]) / params["space_step"]))
        window_x1_idx = int(
            round((float(params["window_pos_x"]) + float(params["window_width"])) / params["space_step"])
        )
        wx0 = window_x0_idx / stride_used
        wx1 = window_x1_idx / stride_used
        window_segment = (wx0, wx1)

    shapes = _build_border_shapes(
        grid_shape,
        border_thickness=border_thickness,
        window_segment=window_segment,
        radiator_rect=radiator_rect,
    )
    x_tick_vals, x_tick_text = _axis_ticks(
        float(params["room_width"]), float(params["space_step"]), stride_used
    )
    y_tick_vals, y_tick_text = _axis_ticks(
        float(params["room_length"]), float(params["space_step"]), stride_used
    )

    title_time = _format_elapsed_time(steps_idx[0] * time_step)
    if len(grids) == 1:
        fig = px.imshow(
            grids[0],
            origin="lower",
            color_continuous_scale="Turbo",
            labels={"x": "X", "y": "Y", "color": "Temperature (°C)"},
            title=f"Heatmap (step {steps_idx[0]}, time {title_time})",
            zmin=vmin,
            zmax=vmax,
        )
        fig.update_traces(
            hovertemplate="Temperature: %{z:.2f}°C<extra></extra>",
            colorbar=dict(title="Temperature (°C)", len=0.8, thickness=14),
        )
        fig.update_layout(
            title=dict(x=0.5),
            xaxis=dict(
                constrain="domain",
                title_text="",
                tickvals=x_tick_vals,
                ticktext=x_tick_text,
                range=[-border_thickness, grid_shape[1] - 1 + border_thickness],
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                title_text="",
                tickvals=y_tick_vals,
                ticktext=y_tick_text,
                range=[-border_thickness, grid_shape[0] - 1 + border_thickness],
            ),
            shapes=shapes,
            width=765,
            height=765,
        )
        return fig

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=grids[0],
                colorscale="Turbo",
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="Temperature (°C)", len=0.8, thickness=14),
                hovertemplate="Temperature: %{z:.2f}°C<extra></extra>",
            )
        ],
        layout=go.Layout(
            title=dict(text=f"Heatmap (step {steps_idx[0]}, time {title_time})", x=0.5),
            xaxis=dict(
                title="",
                constrain="domain",
                tickvals=x_tick_vals,
                ticktext=x_tick_text,
                range=[-border_thickness, grid_shape[1] - 1 + border_thickness],
            ),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                title="",
                tickvals=y_tick_vals,
                ticktext=y_tick_text,
                range=[-border_thickness, grid_shape[0] - 1 + border_thickness],
            ),
            shapes=shapes,
            width=765,
            height=765,
            sliders=[
                dict(
                    active=0,
                    y=-0.15,
                    x=0.1,
                    len=0.85,
                    currentvalue=dict(prefix="Step/time: "),
                    steps=[
                        dict(
                            label=f"{step_idx} ({_format_elapsed_time(step_idx * time_step)})",
                            method="update",
                            args=[
                                {"z": [grid]},
                                {
                                    "title": {
                                        "text": (
                                            "Heatmap (step "
                                            f"{step_idx}, time "
                                            f"{_format_elapsed_time(step_idx * time_step)})"
                                        ),
                                        "x": 0.5,
                                    }
                                },
                            ],
                        )
                        for grid, step_idx in zip(grids, steps_idx)
                    ],
                )
            ],
        ),
    )
    return fig
