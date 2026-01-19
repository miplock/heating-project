# Ten plik udostepnia publiczne API solvera jako pakiet.
"""Public API for the heat equation solver package."""

from pipeline.solver.core import HeatEquationSolver2D
from pipeline.solver.types import (
    Array,
    BoundaryParams,
    Grid,
    Masks,
    PhysicalParams,
    RadiatorParams,
    TimeGrid,
)

__all__ = [
    "Array",
    "BoundaryParams",
    "Grid",
    "HeatEquationSolver2D",
    "Masks",
    "PhysicalParams",
    "RadiatorParams",
    "TimeGrid",
]
