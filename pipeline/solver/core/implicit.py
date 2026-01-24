"""Implicit diffusion system for the heat equation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from scipy.sparse import csc_matrix, diags
    from scipy.sparse.linalg import splu

    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    csc_matrix = None
    diags = None
    splu = None
    _SPARSE_AVAILABLE = False

from pipeline.solver.types import Array, Grid, PhysicalParams, TimeGrid


class ImplicitSystem:
    """Builds and solves the implicit diffusion step."""

    def __init__(self, grid: Grid, time: TimeGrid, phys: PhysicalParams) -> None:
        self.grid = grid
        self.time = time
        self.phys = phys
        # Wspolczynnik 1/h^2, uzywany w Laplasjanie.
        self._inv_h2 = 1.0 / (self.grid.hx * self.grid.hx)
        # Bufor na macierz dla schematu niejawnego.
        self._mat: Optional[Any] = None
        # Zapamietany rozmiar siatki dla cache macierzy.
        self._shape = None
        # Cache dla faktoryzacji rzadkiej macierzy.
        self._lu: Optional[Any] = None
        # Czy korzystamy z rozwiazywania macierzy rzadkiej.
        self._use_sparse = _SPARSE_AVAILABLE

    def step(self, u_bc: Array, src: Array) -> Array:
        """Solve the implicit diffusion step for the interior nodes."""
        mat = self._matrix()
        rhs = self._rhs(u_bc, src)
        if self._use_sparse:
            if self._lu is None:
                raise RuntimeError("Sparse factorization cache is missing.")
            u_int = self._lu.solve(rhs)
        else:
            u_int = np.linalg.solve(mat, rhs)
        return self._merge(u_bc, u_int)

    def _matrix(self) -> Any:
        """Build or reuse the implicit system matrix."""
        ny, nx = self.grid.shape
        if self._mat is not None:
            if self._shape == (ny, nx):
                return self._mat

        # Liczba niewiadomych dla wnetrza siatki.
        ny_int = ny - 2
        nx_int = nx - 2
        n = ny_int * nx_int
        # Wspolczynnik z kroku czasowego.
        lam = self.phys.alpha * self.time.dt * self._inv_h2

        if self._use_sparse:
            main = (1.0 + 4.0 * lam) * np.ones(n, dtype=float)
            off_x = -lam * np.ones(n - 1, dtype=float)
            off_y = -lam * np.ones(n - nx_int, dtype=float)

            # Wylacz polaczenia miedzy wierszami w kierunku x.
            row_starts = np.arange(1, n) % nx_int == 0
            off_x[row_starts] = 0.0
            row_ends = np.arange(n - 1) % nx_int == (nx_int - 1)
            off_x_plus = off_x.copy()
            off_x_plus[row_ends] = 0.0

            mat = diags(
                diagonals=[main, off_x_plus, off_x, off_y, off_y],
                offsets=[0, 1, -1, nx_int, -nx_int],
                format="csc",
            )
            self._lu = splu(mat)
        else:
            mat = np.zeros((n, n), dtype=float)

            def idx(i: int, j: int) -> int:
                # Indeks w wektorze dla komorki (i, j) wnetrza.
                return i * nx_int + j

            for i in range(ny_int):
                for j in range(nx_int):
                    k = idx(i, j)
                    mat[k, k] = 1.0 + 4.0 * lam
                    if i > 0:
                        mat[k, idx(i - 1, j)] = -lam
                    if i < ny_int - 1:
                        mat[k, idx(i + 1, j)] = -lam
                    if j > 0:
                        mat[k, idx(i, j - 1)] = -lam
                    if j < nx_int - 1:
                        mat[k, idx(i, j + 1)] = -lam

        # Cache dla kolejnych krokow.
        self._mat = mat
        self._shape = (ny, nx)
        return mat

    def _rhs(self, u_bc: Array, src: Array) -> Array:
        """Build the right-hand side for the implicit solve."""
        ny, nx = u_bc.shape
        ny_int = ny - 2
        nx_int = nx - 2
        rhs = np.zeros(ny_int * nx_int, dtype=float)
        lam = self.phys.alpha * self.time.dt * self._inv_h2

        for i in range(ny_int):
            for j in range(nx_int):
                ii = i + 1
                jj = j + 1
                k = i * nx_int + j
                rhs[k] = u_bc[ii, jj] + self.time.dt * src[ii, jj]
                # Wplyw warunkow brzegowych dla sasiadow.
                if i == 0:
                    rhs[k] += lam * u_bc[ii - 1, jj]
                if i == ny_int - 1:
                    rhs[k] += lam * u_bc[ii + 1, jj]
                if j == 0:
                    rhs[k] += lam * u_bc[ii, jj - 1]
                if j == nx_int - 1:
                    rhs[k] += lam * u_bc[ii, jj + 1]
        return rhs

    def _merge(self, u_bc: Array, u_int: Array) -> Array:
        """Merge interior solution back into the full grid."""
        ny, nx = u_bc.shape
        ny_int = ny - 2
        nx_int = nx - 2
        out = u_bc.copy()
        # Wstawiamy rozwiazanie wnetrza do pelnej siatki.
        out[1:ny - 1, 1:nx - 1] = u_int.reshape((ny_int, nx_int))
        return out
