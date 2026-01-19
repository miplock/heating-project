# Ten plik definiuje typy i parametry solvera rownania ciepla.
"""Type and parameter definitions for the heat equation solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Grid:
    """Spatial grid definition for the 2D solver."""

    nx: int
    ny: int
    hx: float  # Spatial step (hx == hy).

    @property
    def shape(self) -> Tuple[int, int]:
        """Return array shape as (rows, cols)."""
        return (self.ny, self.nx)


@dataclass(frozen=True)
class TimeGrid:
    """Temporal grid definition for time stepping."""

    dt: float
    t_end: float

    @property
    def n_steps(self) -> int:
        """Return number of time steps needed to reach t_end."""
        return int(np.ceil(self.t_end / self.dt))


@dataclass(frozen=True)
class PhysicalParams:
    """Physical parameters for the heat equation."""

    alpha: float  # Diffusivity in u_t = alpha * Lap(u).


@dataclass(frozen=True)
class BoundaryParams:
    """Boundary condition configuration.

    Supported kinds:
        - "neumann0": insulated boundary (zero normal derivative)
        - "dirichlet": fixed boundary temperature
    """

    kind: str
    dirichlet_value: Optional[float] = None


@dataclass(frozen=True)
class RadiatorParams:
    """Radiator and thermostat configuration.

    The radiator heats only when mean room temperature is below setpoint.
    """

    enabled: bool = True
    setpoint: float = 293.15  # K (about 20°C).
    coeff: float = 0.0  # Effective heating coefficient.


@dataclass(frozen=True)
class Masks:
    """Geometry masks used to limit and annotate the domain."""

    boundary: Optional[Array] = None
    windows: Optional[Array] = None
    walls: Optional[Array] = None
    radiator: Optional[Array] = None
    domain: Optional[Array] = None
