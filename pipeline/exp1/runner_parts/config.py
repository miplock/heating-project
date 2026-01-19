"""Configuration model and builder for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.exp1.experimental_params import resolve_params


@dataclass(frozen=True)
class Exp1Config:
    room_width_m: float
    room_height_m: float
    hx: float
    t_end: float
    dt: float | None
    alpha: float
    u_out_C: float
    u0_C: float
    setpoint_C: float
    radiator_coeff: float
    window_width_m: float
    radiator_size_m: tuple[float, float]
    radiator_center_x_m: float | None
    bc_kind: str


def build_config(*, use_experimental: bool) -> Exp1Config:
    """Load presets and map them to the dataclass."""
    preset = resolve_params(use_experimental=use_experimental)
    return Exp1Config(
        room_width_m=preset.room_width,
        room_height_m=preset.room_height,
        hx=preset.hx,
        t_end=preset.t_end,
        dt=preset.dt,
        alpha=preset.alpha,
        u_out_C=preset.u_out_c,
        u0_C=preset.u0_c,
        setpoint_C=preset.setpoint_c,
        radiator_coeff=preset.rad_coeff,
        window_width_m=preset.window_width,
        radiator_size_m=(preset.rad_width, preset.rad_height),
        radiator_center_x_m=preset.rad_center_x,
        bc_kind=preset.bc_kind,
    )
