"""Headless reproducer for zone-bias behavior.

Spins up a full chunked world, lets drones establish targets for ~30s of
sim-time, then paints a HIGH zone centered on the highest-PoC cell in the
map. Measures per-tick: how many drones are inside the zone, how many are
bidding for operator_high_zone assets, and the mean distance from each
drone to the zone centroid.

If the market is attracting drones we expect:
  - mean_distance_to_zone decreases over time
  - drones_inside_zone trends up
  - at least some drones receive operator_high_zone assignments

If the market is repelling them we'd see the opposite.

Run:
    cd backend && uv run python scripts/zone_bias_probe.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

import numpy as np  # noqa: E402

from src.agents.coordinator import SwarmCoordinator  # noqa: E402
from src.simulation.drone import create_drone_fleet  # noqa: E402
from src.simulation.engine import tick_chunked  # noqa: E402
from src.simulation.search_map import SearchMap  # noqa: E402
from src.simulation.types import (  # noqa: E402
    Command,
    DroneStatus,
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.terrain.chunked import ChunkedWorld  # noqa: E402


def setup_world(seed: int = 42, drone_count: int = 20, world_size: int = 10240):
    # Match main.py production config (not presets.py) — drain 0.02 gives
    # ~83 min nominal endurance, 5.7km round-trip reach from base.
    sim_config = SimConfig(
        terrain_size=1024, drone_count=drone_count, survivor_count=25,
        drone_sensor_range=40.0, drone_comms_range=120.0,
        drone_battery_drain_rate=0.02, transponder_ratio=0.15,
        detection_requires_los=True, tick_rate=10.0,
    )
    chunked_world = ChunkedWorld(world_size, 1024, seed, sim_config)
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
    return world, sim_config, chunked_world


def pick_reachable_hotspot(coord: SwarmCoordinator, world: WorldState, base: Vec3):
    """Find the HIGHEST-PoC reachable hotspot within fresh-battery range.

    diverse_hotspots returns hotspots sorted by value descending. Pick the
    first (= hottest) whose round-trip from base fits in battery. If none,
    fall back to the hottest unconditionally (we want to test zone bias
    on a real peak, not an empty cell).
    """
    sm = world.search_map
    hotspots = sm.diverse_hotspots(50, min_separation_meters=300.0)
    # Prefer hot & reachable. Fresh-drone one-way range is ~800m after
    # safety margin, so round-trip limit = 2×distance_from_base → need dist ≤ ~700m.
    for col, row, val in hotspots:
        if val <= 0.0:
            continue
        x, z = sm.cell_to_world(col, row)
        d = math.hypot(x - base.x, z - base.z)
        if d < 700.0:
            return (x, z, val)
    # Fallback: just the hottest cell, even if out of range — probe will show
    # "drones can't reach" if that's the case.
    for col, row, val in hotspots:
        if val <= 0.0:
            continue
        x, z = sm.cell_to_world(col, row)
        return (x, z, val)
    return None


def count_poc_cells_in_zone(world: WorldState, xmin: float, zmin: float, xmax: float, zmax: float, min_val: float = 0.0) -> tuple[int, float]:
    """Return (count, peak_value) for PoC hotspots whose centroid falls inside the zone."""
    sm = world.search_map
    hotspots = sm.diverse_hotspots(80, min_separation_meters=200.0)
    cnt = 0
    peak = 0.0
    for col, row, val in hotspots:
        if val <= min_val:
            continue
        x, z = sm.cell_to_world(col, row)
        if xmin <= x <= xmax and zmin <= z <= zmax:
            cnt += 1
            peak = max(peak, val)
    return cnt, peak


def drones_inside(world: WorldState, xmin: float, zmin: float, xmax: float, zmax: float) -> int:
    return sum(
        1 for d in world.drones
        if d.status == DroneStatus.ACTIVE and xmin <= d.position.x <= xmax and zmin <= d.position.z <= zmax
    )


def mean_distance_to_centroid(world: WorldState, cx: float, cz: float) -> float:
    actives = [d for d in world.drones if d.status == DroneStatus.ACTIVE]
    if not actives:
        return float("nan")
    return sum(math.hypot(d.position.x - cx, d.position.z - cz) for d in actives) / len(actives)


def main() -> int:
    print("== zone_bias_probe ==")
    world, sim_config, chunked = setup_world(seed=42)
    coord = SwarmCoordinator(sim_config)
    rng = np.random.default_rng(42)
    dt = 1.0 / sim_config.tick_rate

    # Run 30s of sim to let drones establish targets
    warmup_ticks = 300
    print(f"Warming up {warmup_ticks} ticks...")
    for _ in range(warmup_ticks):
        commands = coord.update(world, sim_config)
        world = tick_chunked(world, dt, chunked, commands, rng=rng, config=sim_config)

    # Sample hotspots using the SAME NMS params the coordinator will use
    # inside _run_priority_market, so we pick a zone over cells that are
    # actually in the market's candidate pool.
    sm = world.search_map
    active_count = sum(1 for d in world.drones if d.status == DroneStatus.ACTIVE)
    world_size = max(world.terrain.width, world.terrain.height)
    hotspot_separation = 0.4 * (world_size / math.sqrt(active_count))
    n_spots = active_count * 2 + 5
    market_hotspots = sm.diverse_hotspots(n_spots, min_separation_meters=hotspot_separation)
    # Of the market's hotspots, pick the one closest to base
    candidates = []
    for col, row, val in market_hotspots:
        if val <= 0.0:
            continue
        x, z = sm.cell_to_world(col, row)
        d = math.hypot(x - world.base_position.x, z - world.base_position.z)
        candidates.append((d, x, z, val, col, row))
    if not candidates:
        print("No PoC hotspots at all.")
        return 1
    candidates.sort()
    d_base, cx, cz, poc_val, col0, row0 = candidates[0]
    half = max(hotspot_separation, 500.0)  # make zone big enough to span NMS separation
    xmin, xmax = cx - half, cx + half
    zmin, zmax = cz - half, cz + half
    base_dist = math.hypot(cx - world.base_position.x, cz - world.base_position.z)
    print(f"Base at ({world.base_position.x:.0f}, {world.base_position.z:.0f})")
    print(f"Target zone centroid: ({cx:.0f}, {cz:.0f})  PoC={poc_val:.4f}  dist_to_base={base_dist:.0f}m")
    print(f"Zone bbox: x=[{xmin:.0f}, {xmax:.0f}], z=[{zmin:.0f}, {zmax:.0f}]")
    n_poc, peak = count_poc_cells_in_zone(world, xmin, zmin, xmax, zmax)
    print(f"PoC cells inside zone: {n_poc}  (peak value {peak:.4f})")

    # Snapshot drone positions before painting
    print("\n[BEFORE paint]")
    pre_mean = mean_distance_to_centroid(world, cx, cz)
    pre_inside = drones_inside(world, xmin, zmin, xmax, zmax)
    print(f"  mean drone→zone = {pre_mean:.0f}m  inside = {pre_inside}")
    tasks: dict[str, int] = {}
    batteries: list[float] = []
    for d in world.drones:
        if d.status != DroneStatus.ACTIVE:
            continue
        agent = coord.agents.get(d.id)
        t = agent.task.name if agent else "?"
        tasks[t] = tasks.get(t, 0) + 1
        batteries.append(d.battery)
    print(f"  active task breakdown: {tasks}")
    if batteries:
        print(
            f"  battery range: min={min(batteries):.0f}%  "
            f"mean={sum(batteries)/len(batteries):.0f}%  max={max(batteries):.0f}%"
        )

    # Apply zone_command
    polygon = [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)]
    cmd = Command(
        type="set_priority",
        zone_id="probe_high",
        priority="high",
        data={
            "action": "create",
            "zone_id": "probe_high",
            "priority": "high",
            "polygon": [list(pt) for pt in polygon],
        },
    )
    coord.apply_zone_command(cmd, world)
    print(f"Painted HIGH zone. Backend sees {len(coord.zones)} zone(s).")

    # --- Ground-truth bidding test: force every drone into the market ---
    # Clear all drone targets + sweep waypoints so every active drone is in
    # the bidder pool on the next tick. This isolates "does the market
    # prefer zone assets" from "how fast do drones naturally cycle in".
    print("\n[Forcing full market cycle — clearing all drone targets]")
    for d in world.drones:
        a = coord.agents.get(d.id)
        if a is None:
            continue
        a.current_target = None
        a.local_sweep_waypoints = None
        a.ticks_at_target = 0
    coord._market_tick = -1  # force recompute next tick
    coord._poc_cache_tick = -1

    commands = coord.update(world, sim_config)
    asgn = coord._market_assignments
    by_source: dict[str, int] = {}
    in_zone = 0
    for a in asgn.values():
        by_source[a.source] = by_source.get(a.source, 0) + 1
        if xmin <= a.x <= xmax and zmin <= a.z <= zmax:
            in_zone += 1
    print(f"Market ran with {len(asgn)} drones asking. Assignment sources:")
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")
    print(f"Assignments geographically inside the zone bbox: {in_zone}/{len(asgn)}")

    # Verbose per-drone assignment + bid-against-top
    print("\n[Per-drone assignment]")
    print("  drone  task                  assigned_to (source)    pos → asset (distance)")
    for d in sorted(world.drones, key=lambda x: x.id):
        if d.status != DroneStatus.ACTIVE:
            continue
        a = asgn.get(d.id)
        agent = coord.agents.get(d.id)
        task = agent.task.name if agent else "?"
        if a is None:
            print(f"  {d.id:>3}   {task:<20}  <none>")
            continue
        dist = math.hypot(d.position.x - a.x, d.position.z - a.z)
        in_zone_flag = "★" if (xmin <= a.x <= xmax and zmin <= a.z <= zmax) else " "
        print(f"  {d.id:>3}   {task:<20}  {a.asset_id:<16} ({a.source:<20}){in_zone_flag}  d={dist:.0f}m")
    # Apply tick result back
    world = tick_chunked(world, dt, chunked, commands, rng=rng, config=sim_config)

    # Run another 90s and sample every 10s
    sample_every = 100  # ticks (10s)
    total = 900
    print(f"\n[AFTER paint] tick  mean_to_zone  inside  high_zone_assets  assigned_to_zone")
    for step in range(total):
        commands = coord.update(world, sim_config)
        world = tick_chunked(world, dt, chunked, commands, rng=rng, config=sim_config)
        if (step + 1) % sample_every == 0:
            # Peek at market state
            ops_assets = sum(
                1 for a in coord._market_assignments.values()
                if a.source == "operator_high_zone"
            )
            # Count assets (broader — inspect assignments dict directly)
            n_assign = len(coord._market_assignments)
            mean_d = mean_distance_to_centroid(world, cx, cz)
            inside = drones_inside(world, xmin, zmin, xmax, zmax)
            print(f"          {world.tick:4d}  {mean_d:10.0f}m  {inside:4d}   {ops_assets:4d}  {n_assign:4d} asgn")

    # Post-mortem: what did the market actually assign?
    print("\n== Final market snapshot ==")
    print(f"assignments: {len(coord._market_assignments)}")
    by_source: dict[str, int] = {}
    for a in coord._market_assignments.values():
        by_source[a.source] = by_source.get(a.source, 0) + 1
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")

    # Verdict. The right metric is NOT "mean distance" — with realistic
    # reach, the 16 non-zone drones correctly spread out to cover the rest
    # of the 10km world, pulling the mean up. The right metric is: how
    # many drones the market assigned to an operator_high_zone asset.
    # Real SAR: operator says "prioritize here", a few drones redirect,
    # the rest keep covering the map.
    zone_assignments = sum(
        1 for a in coord._market_assignments.values()
        if a.source == "operator_high_zone"
    )
    inside_final = drones_inside(world, xmin, zmin, xmax, zmax)
    print(f"\nDrones assigned to operator_high_zone at final tick: {zone_assignments}")
    print(f"Drones geographically inside zone bbox: {inside_final}")
    if zone_assignments >= 3:
        print("✓ Market is attracting drones to the zone (good).")
    elif zone_assignments >= 1:
        print("~ Partial pull — zone attracts some drones but fewer than expected.")
    else:
        print("✗ No drones assigned to zone assets — bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
