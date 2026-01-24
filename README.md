# Heating project (modelowanie deterministyczne)

Projekt do symulacji 2D rownania ciepla z prostym modelem grzejnika.
Repo zawiera solver, eksperymenty i notatniki zwiazane z dwoma pytaniami
badawczymi (Q1 i Q2).

## Pytania badawcze
**Q1:** Czy grzejnik musi byc pod oknem?  
**Q2:** Czy wylaczanie grzejnikow podczas nieobecnosci ma sens?

Q1 jest zaimplementowane jako `exp1`. Q2 jest planowane (patrz `TODO.md`).

## Struktura (bez mieszania warstw)
**Solver (model/numerics):**
- `pipeline/solver/` - publiczne API solvera
- `pipeline/solver/core/` - silnik (krok niejawny, warunki brzegowe, zrodlo)
- `pipeline/solver/types.py` - typy i parametry (Grid/TimeGrid/BoundaryParams itd.)
- `pipeline/geometry.py` - narzedzia geometrii i masek
- `pipeline/physical_constants.py` - stale fizyczne

**Eksperymenty:**
- `pipeline/exp1/` - eksperyment Q1 (runner, konfiguracja, parametry, logika metryk)

**Analiza/metryki:**
- `pipeline/metrics.py` - metryki globalne (energia, srednia, odchylenie)
- `pipeline/exp1/metrics_logic.py` - metryki specyficzne dla Q1
- `data/exp1_results.csv` - tabela wynikow Q1

**Wizualizacja:**
- `pipeline/exp1/visualize.py` - skladanie wizualizacji Q1
- `pipeline/exp1/plots/` - wykresy i mapy dla Q1
- `notebooks/*.ipynb` - notatniki pomocnicze/raportowe

**Materialy pomocnicze:**
- `heating-project-v1.pdf` - opis projektu
- `notebooks/results_roadmap.md` - roadmapa wynikow dla Q1/Q2
- `TODO.md` - lista zadan

## Szybki start
Wymagania:
- Python 3.10+
- pakiety z `requirements.txt`

Instalacja:
```bash
pip install -r requirements.txt
```

Przyklad uzycia solvera:
```python
import numpy as np
from pipeline.solver import (
    BoundaryParams,
    Grid,
    HeatEquationSolver2D,
    PhysicalParams,
    TimeGrid,
)

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

Uruchomienie Q1 (exp1):
```bash
python -m pipeline.exp1.runner --progress
```

## Notatniki
Notatniki znajduja sie w `notebooks/`. Uruchom np.:
```bash
jupyter lab
```

## Licencja
Brak (projekt akademicki).
