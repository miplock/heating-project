# Ten plik zawiera glowna klase solvera rownania ciepla.
"""Core solver implementation for the 2D heat equation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pipeline.metrics import EnergyAccumulator
from pipeline.physical_constants import load_physical_constants_safe
from pipeline.solver.types import (
    Array,
    BoundaryParams,
    Grid,
    Masks,
    PhysicalParams,
    RadiatorParams,
    TimeGrid,
)

from .boundary import apply_boundary
from .domain import apply_domain_mask
from .implicit import ImplicitSystem
from .source import source_term
from .validation import validate_u


class HeatEquationSolver2D:
    """Implicit solver for the 2D heat equation on a rectangular grid.

    The base model solves:
        u_t = alpha * Lap(u)

    The scheme uses a backward Euler step for diffusion, with a
    semi-implicit source term evaluated at the previous step.
    """

    def __init__(
        self,
        grid: Grid,
        time: TimeGrid,
        phys: PhysicalParams,
        bc: BoundaryParams,
        masks: Optional[Masks] = None,
        radiator: Optional[RadiatorParams] = None,
    ) -> None:
        # Zapisujemy konfiguracje solvera i opcjonalne maski geometrii.
        self.grid = grid
        self.time = time
        self.phys = phys
        self.bc = bc
        self.masks = masks or Masks()
        self.radiator = radiator or RadiatorParams(enabled=False)

        # Uklad niejawny dla kroku dyfuzji.
        self._implicit = ImplicitSystem(grid=grid, time=time, phys=phys)
        # Stale powietrza do przeliczen zrodla ciepla.
        air = load_physical_constants_safe()
        self._air_cp = air.specific_heat_air_cp
        self._air_r = air.specific_gas_constant_air
        self._air_p = air.atmospheric_pressure

    def simulate(
        self,
        u0: Array,
        *,
        store_every: int = 1,
        return_times: bool = True,
        callback: Optional[Callable[[int, float, Array], None]] = None,
    ) -> Dict[str, Any]:
        """Simulate the heat equation from t=0 to t_end.

        Args:
            u0: Initial temperature field with shape (ny, nx).
            store_every: Snapshot interval in steps.
            return_times: Whether to include times in the output.
            callback: Optional function called after each step.

        Returns:
            Dictionary with snapshots under "U", optional times "t",
            and accumulated energy under "Psi".
        """
        # Walidacja i kopia startowego pola temperatur.
        u = self._validate_u(u0).copy()
        # Lista zapisanych stanow w czasie.
        snapshots = []
        # Lista czasow, jesli chcemy je zwracac.
        times = []

        # Liczba krokow czasowych i krok dt.
        n_steps = self.time.n_steps
        dt = self.time.dt

        # Akumulator energii do diagnostyki.
        acc = EnergyAccumulator(hx=self.grid.hx, dt=self.time.dt)
        # Historia energii do zwrocenia.
        psi_hist = []

        # Petla po krokach czasu, wlacznie z krokiem koncowym.
        for k in range(n_steps + 1):
            # Czas rzeczywisty dla biezacego kroku.
            t = min(k * dt, self.time.t_end)

            # Zapis stanu co store_every krokow.
            if k % store_every == 0:
                snapshots.append(u.copy())
                psi_hist.append(acc.psi)
                if return_times:
                    times.append(t)

            # Nie wykonujemy kroku po osi czasu poza koncem symulacji.
            if k == n_steps:
                break

            # Pojedynczy krok oraz zrodlo do diagnostyki energii.
            u, src = self.step_with_source(u)
            # Aktualizacja akumulatora energii.
            acc = acc.step(src)

            # Opcjonalny callback dla zewnetrznych logow.
            if callback is not None:
                callback(k + 1, t + dt, u)

        # Slownik wynikow, minimalnie lista stanow i energia.
        out: Dict[str, Any] = {"U": snapshots, "Psi": psi_hist}
        # Czas zwracamy tylko gdy uzytkownik tego chce.
        if return_times:
            out["t"] = times
        return out

    def step(self, u: Array) -> Array:
        """Advance the solution by one implicit time step."""
        u = self._validate_u(u)

        # Warunki brzegowe musza byc narzucone przed krokiem w czasie.
        u_bc = self.apply_boundary(u)
        # Zrodlo ciepla liczone z poprzedniego kroku.
        src = self.source_term(u_bc)
        # Niejawny krok dyfuzji z uwzglednieniem zrodla.
        unew = self._implicit_step(u_bc, src)
        # Wymuszenie domeny, jesli maska jest zdefiniowana.
        unew = self.apply_domain_mask(unew, u_bc)

        return unew

    def step_with_source(self, u: Array) -> Tuple[Array, Array]:
        """Advance one step and return both state and source term."""
        # Walidacja wejscia i kopiowanie pola na brzegach.
        u = self._validate_u(u)
        # Warunki brzegowe.
        u_bc = self.apply_boundary(u)
        # Zrodlo ciepla.
        src = self.source_term(u_bc)
        # Niejawny krok dyfuzji.
        unew = self._implicit_step(u_bc, src)
        # Wymuszenie maski domeny.
        unew = self.apply_domain_mask(unew, u_bc)
        return unew, src

    def _implicit_step(self, u_bc: Array, src: Array) -> Array:
        """Solve the implicit diffusion step for the interior nodes."""
        return self._implicit.step(u_bc, src)

    def apply_boundary(self, u: Array) -> Array:
        """Apply boundary conditions on all four edges."""
        return apply_boundary(u, self.bc)

    def source_term(self, u: Array) -> Array:
        """Compute the heat source term for the radiator."""
        return source_term(
            u,
            masks=self.masks,
            radiator=self.radiator,
            air_cp=self._air_cp,
            air_r=self._air_r,
            air_p=self._air_p,
        )

    def apply_domain_mask(self, unew: Array, u_prev: Array) -> Array:
        """Clamp updates outside the domain to the previous values."""
        return apply_domain_mask(unew, u_prev, self.masks)

    def _validate_u(self, u: Array) -> Array:
        """Validate input array shape and values."""
        return validate_u(u, self.grid.shape)
