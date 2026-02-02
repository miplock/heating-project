# Project structure (Python-focused)

This file summarizes which folders contain Python files and what kind of code they hold.

## pipeline/
- `pipeline/shared/`: shared loaders/utilities for common data files (e.g., physical constants).
- `pipeline/exp1/`: 1st experiment-specific core logic (e.g., the heat solver).
- `pipeline/exp1/runner/`: 1st experiment-specific runner scripts and I/O helpers (load parameters, save results, generate CSV).

## Other folders
Currently no `.py` files outside `pipeline/`.
