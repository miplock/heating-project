# Heating project (modelowanie deterministyczne)

Projekt do symulacji 2D rownania ciepla (heat equation) z prostym modelem
zrodla ciepla (grzejnik). Repo zawiera bazowy solver i notatniki do
przeprowadzania eksperymentow z zadania projektowego.

## Struktura
- `pipeline/solver.py` - solver 2D rownania ciepla + konfiguracje siatki i BC
- `pipeline/metrics.py` - proste metryki (energia, srednia i odchylenie temperatury)
- `notebooks/*.ipynb` - notatniki pomocnicze/raportowe
- `heating-project-v1.pdf` - opis projektu (wejscie)
- `TODO.md` - lista zadan

## Szybki start
Wymagania:
- Python 3.10+
- pakiety z `requirements.txt`

Instalacja:
```bash
pip install -r requirements.txt
```

Przyklad uzycia w kodzie:
```python
import numpy as np
from pipeline.solver import Grid, TimeGrid, PhysicalParams, BoundaryParams, HeatEquationSolver2D

ny, nx = 50, 80
hx = 0.05

grid = Grid(nx=nx, ny=ny, hx=hx)
time = TimeGrid(dt=0.01, t_end=1.0)
phys = PhysicalParams(alpha=1.0e-4)
bc = BoundaryParams(kind="dirichlet", dirichlet_value=293.15)

u0 = np.full((ny, nx), 293.15)
solver = HeatEquationSolver2D(grid, time, phys, bc)

out = solver.simulate(u0, store_every=10)
U = out["U"]
```

## Notatniki
Notatniki znajduja sie w `notebooks/`. Uruchom np.:
```bash
jupyter lab
```

## Licencja
Brak (projekt akademicki).
