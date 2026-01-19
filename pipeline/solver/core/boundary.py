"""Boundary condition utilities."""

from __future__ import annotations

from pipeline.solver.types import Array, BoundaryParams


def apply_boundary(u: Array, bc: BoundaryParams) -> Array:
    """Apply boundary conditions on all four edges."""
    # Kopia pola, zeby nie modyfikowac wejscia in-place.
    u2 = u.copy()
    # Wymiary wierszy i kolumn.
    ny, nx = u2.shape

    if bc.kind == "dirichlet":
        if bc.dirichlet_value is None:
            raise ValueError("For 'dirichlet' set dirichlet_value.")
        val = float(bc.dirichlet_value)
        # Ustawienie stalej temperatury na kazdej krawedzi.
        u2[0, :] = val
        u2[ny - 1, :] = val
        u2[:, 0] = val
        u2[:, nx - 1] = val
        return u2

    if bc.kind == "neumann0":
        # Zerowy strumien: kopiujemy wartosci od sasiadow.
        u2[0, :] = u2[1, :]
        u2[ny - 1, :] = u2[ny - 2, :]
        u2[:, 0] = u2[:, 1]
        u2[:, nx - 1] = u2[:, nx - 2]
        return u2

    raise ValueError(f"Unknown BC kind: {bc.kind}")
