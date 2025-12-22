# pipeline/solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Grid:
    """Opis siatki przestrzennej."""
    nx: int
    ny: int
    hx: float  # krok przestrzenny (zakładamy hx = hy)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)  # UWAGA: (rows=y, cols=x)


@dataclass(frozen=True)
class TimeGrid:
    """Opis siatki czasowej."""
    dt: float
    t_end: float

    @property
    def n_steps(self) -> int:
        return int(np.ceil(self.t_end / self.dt))


@dataclass(frozen=True)
class PhysicalParams:
    """Parametry fizyczne modelu."""
    alpha: float  # dyfuzyjność cieplna w równaniu: u_t = alpha * Lap(u)


@dataclass(frozen=True)
class BoundaryParams:
    """
    Parametry warunków brzegowych.
    Na start robimy proste przypadki, a docelowo rozbudujesz o okna/ściany
    z różnymi lambdami.
    """
    kind: str  # "neumann0" (izolacja) albo "dirichlet" (stała temperatura)
    dirichlet_value: Optional[float] = None  # używane tylko przy kind="dirichlet"


@dataclass(frozen=True)
class RadiatorParams:
    """
    Parametry grzejnika + termostatu.

    model: f(x,u) = coeff * u na obszarze radiatora, jeśli mean(room) < setpoint
    """
    enabled: bool = True
    setpoint: float = 293.15  # K (np. 20°C)
    # [1/s] efektywny współczynnik grzania (zastępuje P*r/(p*A*c))
    coeff: float = 0.0


@dataclass(frozen=True)
class Masks:
    """
    Maski elementów geometrii (opcjonalne na start).
    True oznacza punkty należące do danego obiektu.
    """
    # Zewnętrzny brzeg obszaru (ściany zewnętrzne) – może być wyliczany automatycznie,
    # ale wygodnie mieć maskę.
    boundary: Optional[Array] = None

    # Okna / ściany o różnych własnościach – dopniesz później
    windows: Optional[Array] = None
    walls: Optional[Array] = None

    # Grzejnik – dopniesz później
    radiator: Optional[Array] = None

    # Maska „wnętrza” (gdzie liczymy PDE). Jeśli None -> całe pole.
    domain: Optional[Array] = None


class HeatEquationSolver2D:
    """
    Solver 2D dla równania ciepła na prostokątnej siatce.

    Bazowo:
        u_t = alpha * Lap(u)

    Docelowo (projekt):
        u_t = alpha * Lap(u) + f(x,u)   (grzejnik)
        warunki brzegowe typu (okna/ściany) wg bilansu/Fouriera

    Ten solver jest świadomie prosty: ma być stabilną bazą pod eksperymenty.
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
        self.grid = grid
        self.time = time
        self.phys = phys
        self.bc = bc
        self.masks = masks or Masks()
        self.radiator = radiator or RadiatorParams(enabled=False)

        self._inv_h2 = 1.0 / (self.grid.hx * self.grid.hx)

    # -------------------------
    # Public API
    # -------------------------
    def simulate(
        self,
        u0: Array,
        *,
        store_every: int = 1,
        return_times: bool = True,
        callback: Optional[Callable[[int, float, Array], None]] = None,
    ) -> Dict[str, Any]:
        """
        Symulacja od u0 do t_end.

        Parameters
        ----------
        u0 : np.ndarray (ny, nx)
            Początkowy rozkład temperatury.
        store_every : int
            Co ile kroków zapisywać snapshot (1 = każdy krok).
        callback : callable | None
            Funkcja wywoływana po każdym kroku: callback(step, t, u).

        Returns
        -------
        dict z polami:
            "U": lista snapshotów (np.ndarray)
            "t": lista czasów (opcjonalnie)
        """
        u = self._validate_u(u0).copy()
        snapshots = []
        times = []

        n_steps = self.time.n_steps
        dt = self.time.dt

        for k in range(n_steps + 1):
            t = min(k * dt, self.time.t_end)

            if k % store_every == 0:
                snapshots.append(u.copy())
                if return_times:
                    times.append(t)

            if k == n_steps:
                break

            u = self.step(u)

            if callback is not None:
                callback(k + 1, t + dt, u)

        out: Dict[str, Any] = {"U": snapshots}
        if return_times:
            out["t"] = times
        return out

    def step(self, u: Array) -> Array:
        """
        Jeden krok czasowy.

        Na start robimy jawny krok Eulera:
            u^{n+1} = u^n + dt * alpha * Lap(u^n)

        Uwaga: jawny schemat ma ograniczenie stabilności dt <= O(h^2/alpha).
        Docelowo możesz tu podmienić na niejawny (np. solve(A u^{n+1} = b)).
        """
        u = self._validate_u(u)

        # 1) warunki brzegowe (ustaw u na brzegu przed liczeniem Laplasjanu)
        u_bc = self.apply_boundary(u)

        # 2) Laplasjan w środku
        lap = self.laplacian(u_bc)

        # 3) Źródła (np. grzejnik) – na razie 0, potem dopniesz
        src = self.source_term(u_bc)

        # 4) krok czasowy
        unew = u_bc + self.time.dt * (self.phys.alpha * lap + src)

        # 5) opcjonalnie: wymuś domenę (jeśli masz maskę)
        unew = self.apply_domain_mask(unew, u_bc)

        return unew

    # -------------------------
    # Core numerics
    # -------------------------
    def laplacian(self, u: Array) -> Array:
        """
        Laplasjan 2D na siatce (centralne różnice).
        Zwraca tablicę tego samego kształtu.

        Lap(u)[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4 u[i,j]) / h^2
        """
        ny, nx = u.shape
        lap = np.zeros_like(u)

        # wnętrze (bez brzegów)
        lap[1 : ny - 1, 1 : nx - 1] = (
            u[2:ny, 1 : nx - 1]
            + u[0 : ny - 2, 1 : nx - 1]
            + u[1 : ny - 1, 2:nx]
            + u[1 : ny - 1, 0 : nx - 2]
            - 4.0 * u[1 : ny - 1, 1 : nx - 1]
        ) * self._inv_h2

        # jeśli chcesz Laplasjan także na brzegu, to zależy od BC;
        # na start zostawiamy 0 na brzegu (bo i tak BC trzyma wartości)
        return lap

    def apply_boundary(self, u: Array) -> Array:
        """
        Warunki brzegowe globalne.

        Na start:
        - "neumann0": izolacja (pochodna normalna 0) -> kopiowanie wartości z sąsiada
        - "dirichlet": stała temp na brzegu
        """
        u2 = u.copy()
        ny, nx = u2.shape

        if self.bc.kind == "dirichlet":
            if self.bc.dirichlet_value is None:
                raise ValueError("Dla kind='dirichlet' ustaw dirichlet_value.")
            val = float(self.bc.dirichlet_value)
            u2[0, :] = val
            u2[ny - 1, :] = val
            u2[:, 0] = val
            u2[:, nx - 1] = val
            return u2

        if self.bc.kind == "neumann0":
            # górny brzeg: u[0, j] = u[1, j]
            u2[0, :] = u2[1, :]
            # dolny brzeg
            u2[ny - 1, :] = u2[ny - 2, :]
            # lewy brzeg
            u2[:, 0] = u2[:, 1]
            # prawy brzeg
            u2[:, nx - 1] = u2[:, nx - 2]
            return u2

        raise ValueError(f"Nieznany typ BC: {self.bc.kind}")

    def source_term(self, u: Array) -> Array:
        """
        Źródło ciepła f(x,u) wg projektu:
        - grzeje tylko na masce grzejnika
        - tylko gdy średnia temperatura w "pokoju" < setpoint

        Minimalna wersja: pokój = cała domena (albo masks.domain jeśli jest).
        """
        if not self.radiator.enabled:
            return np.zeros_like(u)

        rad = self.masks.radiator
        if rad is None:
            return np.zeros_like(u)

        # Jaka część siatki liczy się jako "pokój" do średniej?
        dom = self.masks.domain
        if dom is None:
            mean_u = float(u.mean())
        else:
            mean_u = float(u[dom].mean())

        if mean_u >= self.radiator.setpoint:
            return np.zeros_like(u)

        src = np.zeros_like(u)
        # Model z PDF: f ~ coeff * u na obszarze grzejnika (multiplikatywnie).
        src[rad] = self.radiator.coeff * u[rad]
        return src

    def apply_domain_mask(self, unew: Array, u_prev: Array) -> Array:
        """
        Jeśli masz maskę domeny (np. nieregularny kształt mieszkania w prostokącie),
        to dla punktów poza domeną możesz:
        - trzymać stałą temp
        - albo kopiować poprzednie wartości
        """
        dom = self.masks.domain
        if dom is None:
            return unew

        out = unew.copy()
        out[~dom] = u_prev[~dom]
        return out

    # -------------------------
    # Helpers
    # -------------------------
    def _validate_u(self, u: Array) -> Array:
        if not isinstance(u, np.ndarray):
            raise TypeError("u musi być np.ndarray.")
        if u.ndim != 2:
            raise ValueError("u musi być tablicą 2D (ny, nx).")
        if u.shape != self.grid.shape:
            raise ValueError(f"Zły rozmiar u: {u.shape}, oczekiwany {self.grid.shape}.")
        if not np.isfinite(u).all():
            raise ValueError("u zawiera NaN/Inf.")
        return u


def stability_suggestion(alpha: float, hx: float) -> float:
    """
    Bardzo zgrubna podpowiedź dla jawnego schematu 2D:
        dt <= h^2 / (4*alpha)
    (to nie jest 'święta' stała, ale pomaga uniknąć eksplozji).
    """
    return (hx * hx) / (4.0 * alpha)
