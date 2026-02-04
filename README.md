# Projekt: Symulacja przewodnictwa ciepla

Repozytorium zawiera eksperymentalny pipeline do symulacji przewodnictwa ciepla (eksperyment `Q1`) wraz z danymi wejsciowymi, skryptami generujacymi wyniki oraz notatnikami do wizualizacji.

## Szybki start
1. Utworz i aktywuj wirtualne srodowisko:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Zainstaluj zaleznosci:
   ```bash
   pip install -r requirements.txt
   ```
3. Uruchom symulacje:
   ```bash
   python -m pipeline.exp1.runner.simulation
   ```

Wyniki zapisuja sie domyslnie do `data/shared/processed/heat_results.pkl`.

## Struktura repozytorium
```
.
├── data/
│   ├── Q1/
│   │   ├── raw/                # parametry symulacji (CSV)
│   │   └── figures/            # zapisane wyniki do wykresow (PKL)
│   └── shared/
│       ├── raw/                # stale fizyczne (CSV)
│       └── processed/          # wyniki symulacji (PKL)
├── notebooks/
│   └── Q1/
│       ├── solver/             # notatniki zwiazane z solverem
│       └── plots/              # notatniki do wizualizacji
├── pipeline/
│   ├── shared/                 # wspolne narzedzia i stale fizyczne
│   └── exp1/                   # logika eksperymentu Q1
├── heating-project-v1.pdf      # opis/zalozenia projektu
├── requirements.txt            # zaleznosci Pythona
└── structure.md                # skrocony opis struktury kodu
```

## Jak dziala pipeline
- `pipeline/shared/physical_constants.py` laduje stale fizyczne z `data/shared/raw/physical_constants.csv`.
- `pipeline/exp1/runner/simulation_parameters.py` laduje parametry symulacji z `data/Q1/raw/sim1_params_def.csv`.
- `pipeline/exp1/solve_heat.py` zawiera solver przewodnictwa ciepla.
- `pipeline/exp1/runner/simulation.py` spina caly przebieg i zapisuje wyniki do `data/shared/processed/heat_results.pkl`.
- Skrypty w `pipeline/*/plots/` i notatniki w `notebooks/` sluza do generowania wykresow.

## Notatniki i wykresy
Notatniki w `notebooks/Q1/plots/` wczytuja zapisane wyniki (PKL) i generuja wykresy, m.in. srednia temperature oraz odchylenie standardowe.

## Dane
Przykladowe dane wejsciowe znajduja sie w `data/Q1/raw/` i `data/shared/raw/`. Wyniki symulacji oraz przetworzone dane do wizualizacji zapisywane sa w `data/shared/processed/` i `data/Q1/figures/`.
