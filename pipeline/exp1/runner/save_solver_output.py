import pickle
from typing import Iterable, Mapping, Optional


def save_solver_output(
    steps: Iterable,
    path: str,
    *,
    simulation_parameters: Optional[Mapping] = None,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """
    Zapisuje strumień obiektów (np. kroki symulacji) do jednego pliku PKL.
    Każdy obiekt jest dumpowany sekwencyjnie.
    """
    with open(path, "wb") as f:
        if simulation_parameters is not None:
            pickle.dump(
                {
                    "type": "metadata",
                    "simulation_parameters": dict(simulation_parameters),
                },
                f,
                protocol=protocol,
            )
        for step in steps:
            pickle.dump(step, f, protocol=protocol)
