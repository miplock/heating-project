# Ten plik obsluguje niedomyslne zestawy parametrow dla exp1.
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional, Any

from pipeline.exp1.params import (
    DEFAULT_PARAMS_PATH,
    Exp1DefaultParams,
    load_params,
)

EXPERIMENTAL_PARAMS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "exp1_params_experimental.json"
)


def load_experimental_params(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load experimental parameter overrides from a JSON file."""
    params_path = path or EXPERIMENTAL_PARAMS_PATH
    if not params_path.exists():
        return {}
    data = json.loads(params_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Experimental params must be a JSON object.")
    allowed = {field.name for field in fields(Exp1DefaultParams)}
    return {key: value for key, value in data.items() if key in allowed}


def resolve_params(
    *,
    use_experimental: bool = False,
    params_path: Optional[Path] = None,
    experimental_path: Optional[Path] = None,
) -> Exp1DefaultParams:
    """Resolve exp1 parameters using defaults and experimental overrides."""
    base = load_params(params_path or DEFAULT_PARAMS_PATH)
    if not use_experimental:
        return base
    overrides = load_experimental_params(experimental_path)
    if not overrides:
        return base
    merged = {**base.__dict__, **overrides}
    return Exp1DefaultParams(**merged)
