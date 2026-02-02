from pathlib import Path
import time

from pipeline.shared.physical_constants import load_physical_constants
from pipeline.exp1.runner.save_solver_output import save_solver_output
from pipeline.exp1.runner.simulation_parameters import load_simulation_parameters
from pipeline.exp1.solve_heat import HeatSolver


def _format_duration(total_seconds: float) -> str:
    seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}min {secs:02d}s"


def _format_progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[?]"
    filled = int(width * current / total)
    return f"[{'=' * filled}{'.' * (width - filled)}]"


def _iter_with_progress(steps, total_steps: int, time_step: float):
    start = time.monotonic()
    last_print = 0.0
    for idx, step in enumerate(steps, start=1):
        yield step
        now = time.monotonic()
        if idx == total_steps or now - last_print >= 0.2:
            elapsed = now - start
            avg = elapsed / idx
            remaining = avg * max(total_steps - idx, 0)
            sim_elapsed = idx * time_step
            bar = _format_progress_bar(idx, total_steps)
            print(
                (
                    f"\r{bar} {idx}/{total_steps}"
                    f" | sim { _format_duration(sim_elapsed)}"
                    f" | ETA { _format_duration(remaining)}"
                ),
                end="",
                flush=True,
            )
            last_print = now
    if total_steps > 0:
        print()


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "pipeline").exists():
            return parent
    return current.parent


CONSTANTS_CSV_PATH = _project_root() / "data/shared/raw/physical_constants.csv"
SIMULATION_PARAMS_CSV_PATH = _project_root() / "data/Q1/raw/sim1_params_def.csv"
RESULTS_PKL_PATH = _project_root() / "data/shared/processed/heat_results.pkl"


def run_simulation(
    constants_csv_path: Path = CONSTANTS_CSV_PATH,
    simulation_params_csv_path: Path = SIMULATION_PARAMS_CSV_PATH,
    results_pkl_path: Path = RESULTS_PKL_PATH,
    progress: bool = False,
) -> None:
    print(f"Loading physical constants from {constants_csv_path}...")
    physical_constants = load_physical_constants(constants_csv_path)
    print(f"Loading simulation parameters from {simulation_params_csv_path}...")
    simulation_parameters = load_simulation_parameters(simulation_params_csv_path)
    print("Initializing solver...")
    solver = HeatSolver(physical_constants, simulation_parameters)
    print(f"Running solver and saving results to {results_pkl_path}...")
    steps_iter = solver.run()
    if progress:
        total_steps = int(simulation_parameters["time_steps"])
        time_step = float(simulation_parameters["time_step"])
        steps_iter = _iter_with_progress(steps_iter, total_steps, time_step)
    save_solver_output(
        steps_iter,
        results_pkl_path,
        simulation_parameters=simulation_parameters,
    )
    print("Done.")


if __name__ == "__main__":
    run_simulation()
