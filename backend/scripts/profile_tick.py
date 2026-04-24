#!/usr/bin/env python3
"""Profile the tick loop to surface real hotspots.

Runs a short headless mission under cProfile and prints the top hottest
functions by cumulative time. Use this before any performance work —
guesses are cheaper than measurements, but worthless.

    cd backend && uv run python -m scripts.profile_tick
    cd backend && uv run python -m scripts.profile_tick --ticks 1200 --top 30
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from io import StringIO

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import numpy as np

from src.agents.coordinator import SwarmCoordinator
from src.simulation.drone import create_drone_fleet
from src.simulation.engine import tick_chunked
from src.simulation.mission import build_mission
from src.simulation.search_map import SearchMap
from src.simulation.types import FOG_UNEXPLORED, SimConfig, Terrain, WorldState
from src.simulation.weather import WeatherSystem
from src.terrain.chunked import ChunkedWorld

WORLD_SIZE = 10240
CHUNK_SIZE = 1024


def _build_world(seed: int, config: SimConfig):
    mission = build_mission("aircraft_crash", WORLD_SIZE, seed)
    cw = ChunkedWorld(WORLD_SIZE, CHUNK_SIZE, seed, config, clusters=mission.clusters)
    base = mission.base_position
    drones = create_drone_fleet(config.drone_count, base, config)
    stub = Terrain(
        width=WORLD_SIZE, height=WORLD_SIZE, max_elevation=config.max_elevation,
        heightmap=cw.make_heightmap_proxy(), biome_map=cw.make_biome_proxy(),
        survivors=(), seed=seed,
    )
    fog_res = max(WORLD_SIZE // 10, 256)
    sm = SearchMap.empty(world_size=float(WORLD_SIZE), cell_size=40.0)
    mission.seed_poc_grid(sm)
    world = WorldState(
        tick=0, elapsed=0.0, terrain=stub, drones=drones, survivors=(),
        fog_grid=np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8),
        comms_links=(), events=(), base_position=base,
        tick_rate=config.tick_rate, search_map=sm,
        evidence=tuple(mission.evidence),
    )
    return world, cw, mission, base


def run(ticks: int, top: int) -> None:
    config = SimConfig(
        terrain_size=CHUNK_SIZE,
        drone_count=20,
        survivor_count=25,
        drone_battery_drain_rate=0.02,
        transponder_ratio=0.15,
    )
    world, cw, _mission, _base = _build_world(seed=42, config=config)
    coordinator = SwarmCoordinator(config)
    weather = WeatherSystem(42, WORLD_SIZE)
    rng = np.random.default_rng(42)
    dt = 1.0 / config.tick_rate

    # Warm up chunks near the base so first-tick chunk gen doesn't dominate.
    for _ in range(5):
        weather.update(world.elapsed)
        commands = coordinator.update(world, config, wind_hazard_fn=weather.is_hazardous_at)
        world = tick_chunked(
            world, dt, cw, commands, rng=rng, config=config,
            wind_fn=weather.get_wind_at,
        )

    pr = cProfile.Profile()
    wall_start = time.monotonic()
    pr.enable()
    for _ in range(ticks):
        weather.update(world.elapsed)
        commands = coordinator.update(world, config, wind_hazard_fn=weather.is_hazardous_at)
        world = tick_chunked(
            world, dt, cw, commands, rng=rng, config=config,
            wind_fn=weather.get_wind_at,
        )
    pr.disable()
    wall = time.monotonic() - wall_start

    sim_seconds = ticks * dt
    print(f"\n{ticks} ticks ({sim_seconds:.1f} sim-seconds) in {wall:.2f}s wall time")
    print(f"Sim speedup: {sim_seconds / wall:.1f}× real-time")
    print(f"Per-tick avg: {wall / ticks * 1000:.2f} ms")
    print()

    buf = StringIO()
    stats = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative")
    stats.print_stats(top)
    print(buf.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    run(args.ticks, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
