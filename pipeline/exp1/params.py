# Ten plik trzyma domyslne parametry dla exp1.
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

from pipeline.physical_constants import load_physical_constants_safe

@dataclass(frozen=True)
class Exp1DefaultParams:
    """Default parameters for exp1 configuration and visualization."""

    room_width: float = 4.0
    room_height: float = 3.0
    hx: float = 0.05
    t_end: float = 50.0
    dt: float = 0.01
    alpha: float = 2.1e-5
    u_out_c: float = 0.0
    u0_c: float = 12.0
    setpoint_c: float = 20.0
    rad_coeff: float = 0.02
    window_width: float = 1.2
    rad_width: float = 1.0
    rad_height: float = 0.3
    rad_center_x: Optional[float] = None
    bc_kind: str = "dirichlet"

    @property
    def radiator_size(self) -> Tuple[float, float]:
        """Return radiator size as (width, height)."""
        return (self.rad_width, self.rad_height)


_PHYS = load_physical_constants_safe()
DEFAULT_PARAMS = Exp1DefaultParams(alpha=_PHYS.thermal_diffusivity_air)

# Domyslna sciezka do nadpisan z notebooka lub edycji recznej.
DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "exp1_params.json"
)


def load_params(path: Optional[Path] = None) -> Exp1DefaultParams:
    """Load exp1 defaults with optional JSON overrides from disk."""
    base = DEFAULT_PARAMS
    params_path = path or DEFAULT_PARAMS_PATH
    if not params_path.exists():
        return base
    data = json.loads(params_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("exp1 params override must be a JSON object.")
    allowed = {field.name for field in fields(Exp1DefaultParams)}
    overrides = {key: value for key, value in data.items() if key in allowed}
    merged = {**base.__dict__, **overrides}
    return Exp1DefaultParams(**merged)
