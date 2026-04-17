"""Headless SAR simulation runner — full chunked world, no frontend, no WebSocket.

Runs a complete mission at maximum speed and produces structured output for
analysis. Useful for:
  - Comparing coordinator strategies (sector sweep vs PoC-based)
  - Measuring time-to-find across different seeds / scenarios
  - Finding optimization opportunities (drones wasting time, PoC stalling)
  - Regression testing search effectiveness

Speed: typically 500-2000x real time on a modern laptop (LLMs disabled).

Usage:
    uv run python scripts/headless_sim.py                       # defaults
    uv run python scripts/headless_sim.py --sim-seconds 600     # 10 min mission
    uv run python scripts/headless_sim.py --drones 40 --seed 7
    uv run python scripts/headless_sim.py --no-llm --output /tmp/sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add backend/src to path so this script can be run from anywhere
_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

import numpy as np  # noqa: E402

from src.agents.coordinator import SwarmCoordinator  # noqa: E402
from src.simulation.drone import create_drone_fleet  # noqa: E402
from src.simulation.engine import tick_chunked  # noqa: E402
from src.simulation.search_map import SearchMap  # noqa: E402
from src.simulation.types import (  # noqa: E402
    ChunkedWorldConfig,
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.terrain.chunked import ChunkCoord, ChunkedWorld  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryEvent:
    tick: int
    elapsed: float
    drone_id: int
    survivor_id: int
    drone_pos: tuple[float, float, float]
    survivor_pos: tuple[float, float, float]
    distance_to_survivor: float

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "elapsed": round(self.elapsed, 1),
            "drone_id": self.drone_id,
            "survivor_id": self.survivor_id,
            "drone_pos": [round(v, 1) for v in self.drone_pos],
            "survivor_pos": [round(v, 1) for v in self.survivor_pos],
            "distance_to_survivor": round(self.distance_to_survivor, 1),
        }


@dataclass
class DroneSnapshot:
    tick: int
    drone_id: int
    pos: tuple[float, float, float]
    battery: float
    status: str
    task: str
    distance_traveled: float  # cumulative since start

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "drone_id": self.drone_id,
            "pos": [round(v, 1) for v in self.pos],
            "battery": round(self.battery, 1),
            "status": self.status,
            "task": self.task,
            "distance_traveled": round(self.distance_traveled, 1),
        }


@dataclass
class CoverageSnapshot:
    tick: int
    elapsed: float
    coverage_pct: float
    found_count: int
    total_survivors: int
    active_drones: int
    avg_battery: float
    poc_total_mass: float | None = None
    poc_peak: float | None = None

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "elapsed": round(self.elapsed, 1),
            "coverage_pct": round(self.coverage_pct, 2),
            "found_count": self.found_count,
            "total_survivors": self.total_survivors,
            "active_drones": self.active_drones,
            "avg_battery": round(self.avg_battery, 1),
            "poc_total_mass": round(self.poc_total_mass, 4) if self.poc_total_mass else None,
            "poc_peak": round(self.poc_peak, 6) if self.poc_peak else None,
        }


@dataclass
class SimReport:
    config: dict
    sim_duration_s: float
    wall_time_s: float
    speedup: float
    total_ticks: int
    total_survivors: int
    found_count: int
    total_discoveries: list[DiscoveryEvent] = field(default_factory=list)
    coverage_snapshots: list[CoverageSnapshot] = field(default_factory=list)
    final_drone_states: list[DroneSnapshot] = field(default_factory=list)
    per_drone_distance: dict[int, float] = field(default_factory=dict)
    per_drone_discoveries: dict[int, int] = field(default_factory=dict)
    time_to_first_discovery_s: float | None = None
    time_to_nth_percentile: dict[str, float] = field(default_factory=dict)  # "25%": t_s, etc.

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "sim_duration_s": round(self.sim_duration_s, 1),
            "wall_time_s": round(self.wall_time_s, 2),
            "speedup": round(self.speedup, 0),
            "total_ticks": self.total_ticks,
            "total_survivors": self.total_survivors,
            "found_count": self.found_count,
            "found_pct": round(100 * self.found_count / max(self.total_survivors, 1), 1),
            "time_to_first_discovery_s": (
                round(self.time_to_first_discovery_s, 1)
                if self.time_to_first_discovery_s is not None else None
            ),
            "time_to_nth_percentile": self.time_to_nth_percentile,
            "discoveries": [e.to_dict() for e in self.total_discoveries],
            "coverage_snapshots": [c.to_dict() for c in self.coverage_snapshots],
            "final_drones": [d.to_dict() for d in self.final_drone_states],
            "per_drone_distance": {str(k): round(v, 1) for k, v in self.per_drone_distance.items()},
            "per_drone_discoveries": {str(k): v for k, v in self.per_drone_discoveries.items()},
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_sim(
    sim_seconds: float = 300.0,
    drone_count: int = 20,
    survivor_count: int = 25,
    world_size: int = 10240,
    chunk_size: int = 1024,
    seed: int = 42,
    snapshot_interval_s: float = 10.0,
    use_poc: bool = True,
    verbose: bool = True,
) -> SimReport:
    """Run a full SAR mission headless and return structured report."""
    # Disable LLMs if no API keys (they'd be slow anyway)
    llm_enabled = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MISTRAL_API_KEY"))

    sim_config = SimConfig(
        terrain_size=chunk_size,
        drone_count=drone_count,
        survivor_count=survivor_count,
        drone_sensor_range=40.0,
        drone_comms_range=120.0,
        drone_battery_drain_rate=0.12,
        transponder_ratio=0.15,
        detection_requires_los=True,
        tick_rate=10.0,
    )
    chunked_config = ChunkedWorldConfig(world_size=world_size, chunk_size=chunk_size, seed=seed)

    if verbose:
        print(f"=== Headless SAR Simulation ===")
        print(f"World: {world_size}m × {world_size}m, {chunk_size}m chunks, seed={seed}")
        print(f"Drones: {drone_count}, Survivors: {survivor_count}")
        print(f"Duration: {sim_seconds}s sim time ({int(sim_seconds * sim_config.tick_rate)} ticks)")
        print(f"PoC-based routing: {'ON' if use_poc else 'OFF'}")
        print(f"LLM calls: {'ON' if llm_enabled else 'OFF (no API keys)'}")
        print()

    # World setup (matches main.py)
    t0 = time.monotonic()
    chunked_world = ChunkedWorld(world_size, chunk_size, seed, sim_config)
    base = Vec3(world_size * 0.15, 0.0, world_size * 0.15)
    drones = create_drone_fleet(drone_count, base, sim_config)
    stub_terrain = Terrain(
        width=world_size, height=world_size, max_elevation=sim_config.max_elevation,
        heightmap=chunked_world.make_heightmap_proxy(),
        biome_map=chunked_world.make_biome_proxy(),
        survivors=(), seed=seed,
    )
    fog_res = max(world_size // 10, 256)
    global_fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)

    # Search map with survivor-cluster prior
    search_map = None
    if use_poc:
        search_map = SearchMap.empty(world_size=float(world_size), cell_size=40.0)
        for (cx, cz, cr, cw_w) in chunked_world._survivor_clusters:
            search_map.add_gaussian((cx, cz), radius_meters=max(cr, 200.0), weight=cw_w)
        search_map.poc += np.float32(0.001)
        search_map.normalize(target_mass=1.0)

    world = WorldState(
        tick=0, elapsed=0.0, terrain=stub_terrain,
        drones=drones, survivors=(),
        fog_grid=global_fog, comms_links=(), events=(),
        base_position=base, tick_rate=sim_config.tick_rate,
        search_map=search_map,
    )

    rng = np.random.default_rng(seed)
    dt = 1.0 / sim_config.tick_rate
    coordinator = SwarmCoordinator(sim_config)

    # Count total survivors (ground truth across all chunks)
    all_survivors = chunked_world.get_all_survivors()
    total_survivors = len(all_survivors)

    init_time = time.monotonic() - t0
    if verbose:
        print(f"Init: {init_time:.2f}s ({total_survivors} survivors placed)")
        print()

    # Run the sim
    max_ticks = int(sim_seconds * sim_config.tick_rate)
    snapshot_tick_interval = int(snapshot_interval_s * sim_config.tick_rate)
    report = SimReport(
        config={
            "sim_seconds": sim_seconds, "drone_count": drone_count,
            "survivor_count": survivor_count, "world_size": world_size,
            "seed": seed, "use_poc": use_poc,
        },
        sim_duration_s=0.0, wall_time_s=0.0, speedup=0.0,
        total_ticks=0, total_survivors=total_survivors, found_count=0,
    )

    # Track per-drone distance traveled
    per_drone_distance: dict[int, float] = {d.id: 0.0 for d in drones}
    per_drone_discoveries: dict[int, int] = {d.id: 0 for d in drones}
    last_positions: dict[int, Vec3] = {d.id: d.position for d in drones}
    # Track cumulative discoveries (survivor ids) — world.survivors only has
    # active-chunk survivors at any moment, so it loses discovered survivors
    # when drones leave the area.
    discovered_ids: set[int] = set()

    sim_start = time.monotonic()
    seen_events: set[tuple[int, int]] = set()  # (tick, survivor_id) dedup

    for _ in range(max_ticks):
        # Coordinator decisions (classical only if no LLM)
        commands = coordinator.update(world, sim_config)

        # Tick the world
        world = tick_chunked(world, dt, chunked_world, commands, rng=rng, config=sim_config)

        # Track distance
        for d in world.drones:
            prev = last_positions.get(d.id)
            if prev is not None:
                dx = d.position.x - prev.x
                dz = d.position.z - prev.z
                per_drone_distance[d.id] += math.sqrt(dx * dx + dz * dz)
            last_positions[d.id] = d.position

        # Process events
        for ev in world.events:
            if ev.type.name == "SURVIVOR_FOUND" and ev.survivor_id is not None:
                key = (world.tick, ev.survivor_id)
                if key in seen_events:
                    continue
                seen_events.add(key)

                drone = next((d for d in world.drones if d.id == ev.drone_id), None)
                survivor = next((s for s in world.survivors if s.id == ev.survivor_id), None)
                if drone and survivor:
                    dist = (drone.position - survivor.position).length_xz()
                    evt = DiscoveryEvent(
                        tick=world.tick, elapsed=world.elapsed,
                        drone_id=ev.drone_id, survivor_id=ev.survivor_id,
                        drone_pos=(drone.position.x, drone.position.y, drone.position.z),
                        survivor_pos=(survivor.position.x, survivor.position.y, survivor.position.z),
                        distance_to_survivor=dist,
                    )
                    report.total_discoveries.append(evt)
                    per_drone_discoveries[ev.drone_id] = per_drone_discoveries.get(ev.drone_id, 0) + 1
                    discovered_ids.add(ev.survivor_id)

                    if report.time_to_first_discovery_s is None:
                        report.time_to_first_discovery_s = world.elapsed

        # Snapshots at interval
        if world.tick % snapshot_tick_interval == 0:
            found = len(discovered_ids)  # cumulative, not current-tick
            # Coverage: fraction of global fog grid explored
            explored = int(np.count_nonzero(world.fog_grid != FOG_UNEXPLORED))
            total_cells = int(world.fog_grid.size)
            coverage_pct = 100.0 * explored / max(total_cells, 1)
            active = sum(1 for d in world.drones if d.status.name == "ACTIVE")
            avg_bat = sum(d.battery for d in world.drones) / len(world.drones)
            poc_mass = None
            poc_peak = None
            if world.search_map is not None:
                sm = world.search_map
                poc_mass = float(sm.poc.sum())
                poc_peak = float(sm.poc.max())
            snap = CoverageSnapshot(
                tick=world.tick, elapsed=world.elapsed,
                coverage_pct=coverage_pct, found_count=found,
                total_survivors=total_survivors, active_drones=active,
                avg_battery=avg_bat, poc_total_mass=poc_mass, poc_peak=poc_peak,
            )
            report.coverage_snapshots.append(snap)
            if verbose:
                print(
                    f"  t={world.elapsed:6.1f}s | Cov: {coverage_pct:5.1f}% | "
                    f"Found: {found:2d}/{total_survivors} | "
                    f"Active: {active}/{len(world.drones)} | "
                    f"Bat: {avg_bat:4.0f}% | "
                    f"PoC: mass={poc_mass:.3f} peak={poc_peak:.5f}"
                    if poc_mass is not None else
                    f"  t={world.elapsed:6.1f}s | Cov: {coverage_pct:5.1f}% | "
                    f"Found: {found:2d}/{total_survivors} | Active: {active}/{len(world.drones)} | Bat: {avg_bat:4.0f}%"
                )

    sim_wall_time = time.monotonic() - sim_start

    # Final snapshots per drone
    for d in world.drones:
        report.final_drone_states.append(DroneSnapshot(
            tick=world.tick, drone_id=d.id,
            pos=(d.position.x, d.position.y, d.position.z),
            battery=d.battery, status=d.status.name,
            task=d.current_task,
            distance_traveled=per_drone_distance.get(d.id, 0.0),
        ))

    report.sim_duration_s = world.elapsed
    report.wall_time_s = sim_wall_time
    report.speedup = world.elapsed / max(sim_wall_time, 1e-6)
    report.total_ticks = world.tick
    report.found_count = len(discovered_ids)
    report.per_drone_distance = per_drone_distance
    report.per_drone_discoveries = per_drone_discoveries

    # Time to find N% of survivors
    if report.total_discoveries:
        times = sorted(e.elapsed for e in report.total_discoveries)
        for pct, label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%"), (1.0, "100%")]:
            idx = int(len(times) * pct) - 1
            if 0 <= idx < len(times):
                report.time_to_nth_percentile[label] = round(times[idx], 1)

    if verbose:
        print()
        print("=== Results ===")
        print(f"Simulated: {report.sim_duration_s:.0f}s | Wall time: {sim_wall_time:.2f}s | "
              f"Speedup: {report.speedup:.0f}x")
        print(f"Found: {report.found_count}/{total_survivors} "
              f"({100*report.found_count/max(total_survivors,1):.0f}%)")
        if report.time_to_first_discovery_s is not None:
            print(f"Time to first find: {report.time_to_first_discovery_s:.0f}s")
        for label, t in report.time_to_nth_percentile.items():
            print(f"Time to {label} found: {t:.0f}s")
        print()
        print("Per-drone summary:")
        active = sum(1 for d in world.drones if d.status.name == "ACTIVE")
        failed = sum(1 for d in world.drones if d.status.name == "FAILED")
        print(f"  Active: {active}, Failed: {failed}")
        print(f"  Avg distance traveled: {np.mean(list(per_drone_distance.values())):.0f}m")
        top_finders = sorted(per_drone_discoveries.items(), key=lambda x: -x[1])[:5]
        if top_finders:
            print(f"  Top finders: {top_finders}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Headless SAR simulation runner")
    p.add_argument("--sim-seconds", type=float, default=300.0, help="Mission duration in sim-seconds")
    p.add_argument("--drones", type=int, default=20, help="Number of drones")
    p.add_argument("--survivors", type=int, default=25, help="Target survivor count (placed in clusters)")
    p.add_argument("--world-size", type=int, default=10240, help="World size in meters (square)")
    p.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in meters")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--no-poc", action="store_true", help="Disable PoC-based routing (use sector sweep)")
    p.add_argument("--snapshot-interval", type=float, default=10.0, help="Snapshot interval in sim-seconds")
    p.add_argument("--output", type=str, default=None, help="Write JSON report to this path")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = p.parse_args()

    report = run_sim(
        sim_seconds=args.sim_seconds,
        drone_count=args.drones,
        survivor_count=args.survivors,
        world_size=args.world_size,
        chunk_size=args.chunk_size,
        seed=args.seed,
        snapshot_interval_s=args.snapshot_interval,
        use_poc=not args.no_poc,
        verbose=not args.quiet,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport written: {args.output}")


if __name__ == "__main__":
    main()
