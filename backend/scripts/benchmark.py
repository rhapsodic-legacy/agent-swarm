#!/usr/bin/env python3
"""Headless benchmark harness for the drone swarm simulator.

Runs one or more missions in a tight loop (no WebSocket, no frontend)
for a fixed sim-duration and emits a structured JSON scorecard per
mission. The scorecard is the canonical artifact for comparing runs
across commits — dump it, diff it, plot it.

Usage:
    # Run one mission for 10 sim-minutes, print scorecard to stdout:
    cd backend && uv run python scripts/benchmark.py --mission lost_hiker --duration 600

    # Run all five missions, write scorecards to JSON:
    cd backend && uv run python scripts/benchmark.py --all --duration 600 --out benchmarks/baseline.json

    # Compare two seeds of the same mission:
    cd backend && uv run python scripts/benchmark.py --mission aircraft_crash --seeds 1,2,3 --duration 300

Each scorecard includes:
  * Survivor metrics: found/placed, MTTD, survival-window hit rate
  * Evidence metrics: discovered/planted, time-to-first-evidence,
    evidence→survivor latency (the Phase 3 headline number)
  * Search quality: PoC entropy drop %, coverage %
  * Resource use: total drone-km, active fraction, finds per km
  * Event totals

By default the benchmark uses the same config as the live server so
numbers from here are directly comparable to numbers from a browser run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as wall_time
from pathlib import Path

# Make `src.*` importable regardless of cwd (script lives in backend/scripts).
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import numpy as np

from src.agents.coordinator import SwarmCoordinator
from src.simulation.engine import tick_chunked
from src.simulation.metrics import MetricsTracker
from src.simulation.mission import available_missions, build_mission
from src.simulation.types import (
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    WorldState,
)
from src.terrain.chunked import ChunkedWorld


# Mirror server defaults so benchmark numbers match live-run numbers.
WORLD_SIZE = 10240
CHUNK_SIZE = 1024

DEFAULT_CONFIG = SimConfig(
    terrain_size=CHUNK_SIZE,
    drone_count=20,
    survivor_count=25,
    drone_sensor_range=40.0,
    drone_comms_range=120.0,
    drone_battery_drain_rate=0.02,
    detection_requires_los=True,
    canopy_occlusion=0.7,
    urban_occlusion=0.5,
    weather_visibility=1.0,
    night_penalty=0.4,
    transponder_range=200.0,
    transponder_ratio=0.15,
)


def run_mission(
    mission_name: str,
    seed: int,
    duration_seconds: float,
    config: SimConfig = DEFAULT_CONFIG,
    quiet: bool = False,
) -> dict:
    """Run a single mission headless for `duration_seconds` sim-time.

    Returns the mission's scorecard dict.
    """
    from src.simulation.drone import create_drone_fleet
    from src.simulation.search_map import SearchMap

    mission = build_mission(mission_name, WORLD_SIZE, seed)
    cw = ChunkedWorld(WORLD_SIZE, CHUNK_SIZE, seed, config, clusters=mission.clusters)

    base = mission.base_position
    drones = create_drone_fleet(config.drone_count, base, config)
    stub_terrain = Terrain(
        width=WORLD_SIZE,
        height=WORLD_SIZE,
        max_elevation=config.max_elevation,
        heightmap=cw.make_heightmap_proxy(),
        biome_map=cw.make_biome_proxy(),
        survivors=(),
        seed=seed,
    )
    fog_res = max(WORLD_SIZE // 10, 256)
    fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)

    # Bayesian search map — seeded by the mission, same as the server.
    sm = SearchMap.empty(world_size=float(WORLD_SIZE), cell_size=40.0)
    mission.seed_poc_grid(sm)

    world = WorldState(
        tick=0,
        elapsed=0.0,
        terrain=stub_terrain,
        drones=drones,
        survivors=(),
        fog_grid=fog,
        comms_links=(),
        events=(),
        base_position=base,
        tick_rate=config.tick_rate,
        search_map=sm,
        evidence=tuple(mission.evidence),
    )

    rng = np.random.default_rng(seed)
    dt = 1.0 / config.tick_rate
    coordinator = SwarmCoordinator(config)
    metrics = MetricsTracker(mission=mission)

    target_ticks = int(duration_seconds / dt)

    if not quiet:
        print(
            f"  Running {mission_name} seed={seed} for {duration_seconds:.0f}s "
            f"({target_ticks} ticks)...",
            flush=True,
        )

    wall_start = wall_time.monotonic()
    for _ in range(target_ticks):
        agent_commands = coordinator.update(world, config)
        world = tick_chunked(world, dt, cw, agent_commands, rng=rng, config=config)
        metrics.record_tick(world)

    wall_elapsed = wall_time.monotonic() - wall_start

    card = metrics.scorecard()
    card["wall_clock_seconds"] = round(wall_elapsed, 2)
    card["sim_speedup"] = (
        round(duration_seconds / wall_elapsed, 1) if wall_elapsed > 0 else None
    )
    return card


def print_scorecard(card: dict) -> None:
    """Human-readable one-screen summary of a scorecard."""
    s = card["survivors"]
    e = card["evidence"]
    q = card["search_quality"]
    r = card["resources"]

    print(f"\n  === {card['mission']} (seed {card['seed']}) ===")
    print(
        f"  elapsed={card['elapsed']:.0f}s "
        f"wall={card['wall_clock_seconds']:.1f}s "
        f"(x{card['sim_speedup']} real-time)"
    )
    print(
        f"  survivors: found={s['found']}/{s['total_placed']}  "
        f"in_window={s['found_in_survival_window']}"
        f" ({s['survival_window_pct']}%)  "
        f"MTTD={_fmt_time(s['mttd_seconds'])}  "
        f"first={_fmt_time(s['time_to_first'])}  "
        f"last={_fmt_time(s['time_to_last'])}"
    )
    print(
        f"  evidence:  discovered={e['discovered']}/{e['planted']}  "
        f"first={_fmt_time(e['time_to_first'])}  "
        f"→survivor_latency={_fmt_time(e['evidence_to_survivor_latency'])}"
    )
    print(
        f"  quality:   entropy_drop={q['entropy_drop_pct']}%  "
        f"coverage={q['coverage_pct']}%"
    )
    print(
        f"  resources: {r['total_drone_km']:.1f}km  "
        f"active={r['active_fraction']}  "
        f"m/find={r['meters_per_find']}"
    )
    ev = card.get("events", {})
    summary_events = {
        k: v for k, v in ev.items()
        if k in {"survivor_found", "evidence_found", "drone_failed",
                 "drone_battery_critical", "drone_returned"}
    }
    if summary_events:
        items = " ".join(f"{k}={v}" for k, v in sorted(summary_events.items()))
        print(f"  events:    {items}")


def _fmt_time(t: float | None) -> str:
    if t is None:
        return "--"
    if t < 60:
        return f"{t:.0f}s"
    m = int(t // 60)
    s = int(t % 60)
    return f"{m}m{s:02d}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless benchmark for the drone swarm simulator.",
    )
    parser.add_argument(
        "--mission", default="aircraft_crash",
        help=f"Mission name (default: aircraft_crash). Available: {available_missions()}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all registered missions. Overrides --mission.",
    )
    parser.add_argument(
        "--seeds", default="42",
        help="Comma-separated list of seeds to run (default: 42)",
    )
    parser.add_argument(
        "--duration", type=float, default=600.0,
        help="Sim-seconds to run each mission (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional JSON output path. If omitted, prints scorecards only.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-tick progress lines.",
    )
    args = parser.parse_args()

    mission_names = available_missions() if args.all else [args.mission]
    for m in mission_names:
        if m not in available_missions():
            print(f"Unknown mission: {m}", file=sys.stderr)
            return 2

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_cards: list[dict] = []
    for mname in mission_names:
        for seed in seeds:
            card = run_mission(
                mname, seed, args.duration, quiet=args.quiet,
            )
            all_cards.append(card)
            print_scorecard(card)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "duration_seconds": args.duration,
            "world_size": WORLD_SIZE,
            "scorecards": all_cards,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {len(all_cards)} scorecard(s) to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
