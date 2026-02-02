from __future__ import annotations

import csv
from pathlib import Path


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pipeline").exists():
            return parent
    return current.parent


SIMULATION_PARAMS_DIR = _project_root() / "data/Q1/raw"


def write_simulation_parameters_csv(
    temp_initial: float,
    time_steps: int,
    time_step: float,
    space_step: float,
    room_length: float,
    room_width: float,
    temp_external: float,
    radiator_size_x: float,
    radiator_size_y: float,
    radiator_pos_x: float,
    radiator_pos_y: float,
    window_pos_x: float,
    window_width: float,
    use_iterative_solver: bool = False,
    name_suffix: str = "def",
    output_dir: Path = SIMULATION_PARAMS_DIR,
) -> Path:
    def _is_on_grid(value: float, step: float, *, tol: float = 1e-9) -> bool:
        if step <= 0:
            return False
        k = round(value / step)
        return abs(value - k * step) <= tol

    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    if space_step <= 0:
        raise ValueError("space_step must be positive")
    if room_length <= 0 or room_width <= 0:
        raise ValueError("room_length and room_width must be positive")
    if radiator_size_x <= 0 or radiator_size_y <= 0:
        raise ValueError("radiator_size_x and radiator_size_y must be positive")
    if not (0 <= radiator_pos_x <= room_width and 0 <= radiator_pos_y <= room_length):
        raise ValueError("radiator position must be within room bounds")
    if window_width <= 0:
        raise ValueError("window_width must be positive")
    if not (0 <= window_pos_x <= room_width):
        raise ValueError("window position must be within room bounds")
    if window_pos_x + window_width > room_width:
        raise ValueError("window exceeds room width at given x position")
    if radiator_pos_x + radiator_size_x > room_width:
        raise ValueError("radiator exceeds room width at given x position")
    if radiator_pos_y + radiator_size_y > room_length:
        raise ValueError("radiator exceeds room length at given y position")
    if radiator_pos_x <= 0 or radiator_pos_y <= 0:
        raise ValueError("radiator must not touch room boundaries")
    if radiator_pos_x + radiator_size_x >= room_width:
        raise ValueError("radiator must not touch room boundaries")
    if radiator_pos_y + radiator_size_y >= room_length:
        raise ValueError("radiator must not touch room boundaries")
    if not _is_on_grid(radiator_pos_x, space_step):
        raise ValueError("radiator_pos_x must be aligned to space_step grid")
    if not _is_on_grid(radiator_pos_y, space_step):
        raise ValueError("radiator_pos_y must be aligned to space_step grid")
    if not _is_on_grid(radiator_pos_x + radiator_size_x, space_step):
        raise ValueError("radiator end x must be aligned to space_step grid")
    if not _is_on_grid(radiator_pos_y + radiator_size_y, space_step):
        raise ValueError("radiator end y must be aligned to space_step grid")
    if not _is_on_grid(window_pos_x, space_step):
        raise ValueError("window_pos_x must be aligned to space_step grid")
    if not _is_on_grid(window_width, space_step):
        raise ValueError("window_width must be aligned to space_step grid")

    csv_path = output_dir / f"sim1_params_{name_suffix}.csv"
    rows = [
        {
            "name": "temp_initial",
            "value": temp_initial,
            "units": "K",
            "description": "Initial temperature",
        },
        {
            "name": "time_steps",
            "value": time_steps,
            "units": "count",
            "description": "Number of simulation time steps",
        },
        {
            "name": "time_step",
            "value": time_step,
            "units": "s",
            "description": "Time step size",
        },
        {
            "name": "use_iterative_solver",
            "value": "true" if use_iterative_solver else "false",
            "units": "bool",
            "description": "Use iterative solver",
        },
        {
            "name": "space_step",
            "value": space_step,
            "units": "m",
            "description": "Spatial step size",
        },
        {
            "name": "room_length",
            "value": room_length,
            "units": "m",
            "description": "Room length",
        },
        {
            "name": "room_width",
            "value": room_width,
            "units": "m",
            "description": "Room width",
        },
        {
            "name": "temp_external",
            "value": temp_external,
            "units": "K",
            "description": "External temperature",
        },
        {
            "name": "radiator_size_x",
            "value": radiator_size_x,
            "units": "m",
            "description": "Radiator size in x direction",
        },
        {
            "name": "radiator_size_y",
            "value": radiator_size_y,
            "units": "m",
            "description": "Radiator size in y direction",
        },
        {
            "name": "radiator_pos_x",
            "value": radiator_pos_x,
            "units": "m",
            "description": "Radiator position (x)",
        },
        {
            "name": "radiator_pos_y",
            "value": radiator_pos_y,
            "units": "m",
            "description": "Radiator position (y)",
        },
        {
            "name": "window_pos_x",
            "value": window_pos_x,
            "units": "m",
            "description": "Window position (x)",
        },
        {
            "name": "window_width",
            "value": window_width,
            "units": "m",
            "description": "Window width",
        },
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Simulation parameters CSV saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    write_simulation_parameters_csv(
        temp_initial=100,
        time_steps=10,
        time_step=1.0,
        space_step=0.1,
        room_length=5.0,
        room_width=4.0,
        temp_external=0.0,
        radiator_size_x=1.0,
        radiator_size_y=0.5,
        radiator_pos_x=0.5,
        radiator_pos_y=0.5,
        window_pos_x=3.0,
        window_width=1.0,
        use_iterative_solver=False,
    )
