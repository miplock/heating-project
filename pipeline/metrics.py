# pipeline/metrics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


def K_to_C(uK: Array) -> Array:
    """Kelwiny -> Celsjusze."""
    return uK - 273.15


def mean_temp_C(uK: Array, mask: Array | None = None) -> float:
    """Średnia temperatura (°C) na masce lub na całej siatce."""
    uC = K_to_C(uK)
    if mask is None:
        return float(np.mean(uC))
    return float(np.mean(uC[mask]))


def std_temp_C(uK: Array, mask: Array | None = None) -> float:
    """Odchylenie standardowe temperatury (°C) na masce lub na całej siatce."""
    uC = K_to_C(uK)
    if mask is None:
        return float(np.std(uC))
    return float(np.std(uC[mask]))


@dataclass(frozen=True)
class EnergyAccumulator:
    """
    Dyskretny odpowiednik:
        Ψ(t) = ∫_0^t ∫_Ω f(x,u(x,s)) dx ds

    W dyskretyzacji:
        Psi += sum(src) * hx^2 * dt
    """
    hx: float
    dt: float
    psi: float = 0.0

    def step(self, src: Array) -> "EnergyAccumulator":
        dV = self.hx * self.hx
        dpsi = float(np.sum(src) * dV * self.dt)
        return EnergyAccumulator(hx=self.hx, dt=self.dt, psi=self.psi + dpsi)
