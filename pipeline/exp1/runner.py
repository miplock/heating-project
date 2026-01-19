from __future__ import annotations

"""Experiment 1 runner.

Ten plik ma byc prosty: krok po kroku pokazuje caly przebieg eksperymentu.
Wszystkie detale zostaly przeniesione do runner_parts/.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from pipeline.exp1.runner_parts import (
    allowed_r_range,
    build_config,
    build_r_values,
    grid_from_room,
    parse_args,
    run_exp1,
    save_maps,
    write_csv,
)


def main() -> None:
    # 1) Wczytaj argumenty z linii polecen i wybierz zestaw parametrow.
    args = parse_args()
    config = build_config(use_experimental=args.use_experimental)

    # 2) Zbuduj siatke i zakres dopuszczalnych odleglosci grzejnika.
    grid = grid_from_room(config.room_width_m, config.room_height_m, config.hx)
    r_min, r_max = allowed_r_range(grid, config.radiator_size_m[1])

    # 3) Wyznacz liste przesuniec r (z podanych wartosci lub z zakresu).
    r_values = build_r_values(
        args.r_values,
        args.r_min if args.r_min is not None else r_min,
        args.r_max if args.r_max is not None else r_max,
        args.r_count,
    )

    # 4) Uruchom symulacje dla kazdej wartosci r i zbierz statystyki.
    results, u_time_by_r, t_values = run_exp1(
        config,
        r_values,
        progress=args.progress,
        collect_maps=not args.no_maps,
    )

    # 5) Zapisz podsumowanie do CSV.
    write_csv(results, args.output)

    # 6) Jesli trzeba, zapisz tez pelne mapy temperatur do .npz.
    if u_time_by_r is not None:
        if t_values is None:
            raise ValueError("Missing time values for maps output.")
        save_maps(
            output_path=args.maps_out,
            r_values=r_values,
            t_values=t_values,
            u_time_by_r=u_time_by_r,
        )

    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
