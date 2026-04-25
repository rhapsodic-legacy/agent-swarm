"""Simulation run test — runs the actual sim for N ticks and prints
drone and survivor positions so you can SEE what's happening.

This is NOT a pass/fail unit test. It's a diagnostic that proves
drones move, survivors exist, and detection works by printing
real coordinates every few ticks.

Run with: uv run pytest tests/test_simulation_run.py -x -v -s
"""

from __future__ import annotations

import math

import numpy as np

from src.simulation.drone import create_drone_fleet
from src.simulation.engine import tick_chunked
from src.simulation.types import (
    FOG_UNEXPLORED,
    Command,
    DroneStatus,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.terrain.chunked import ChunkCoord, ChunkedWorld


def _setup():
    config = SimConfig(
        terrain_size=1024,
        drone_count=10,
        survivor_count=25,
        drone_sensor_range=40.0,
        drone_comms_range=120.0,
        drone_battery_drain_rate=0.12,
        transponder_ratio=0.15,
        transponder_range=200.0,
        detection_requires_los=True,
    )
    cw = ChunkedWorld(10240, 1024, 42, config)

    base = Vec3(10240 * 0.15, 0.0, 10240 * 0.15)

    # Pre-generate all chunks so get_all_survivors returns correct positions
    for cz in range(cw._chunks_z):
        for cx in range(cw._chunks_x):
            cw.get_chunk(ChunkCoord(cx, cz))

    drones = create_drone_fleet(10, base, config)
    stub = Terrain(
        width=10240,
        height=10240,
        max_elevation=200.0,
        heightmap=cw.make_heightmap_proxy(),
        biome_map=cw.make_biome_proxy(),
        survivors=(),
        seed=42,
    )
    fog = np.full((1024, 1024), FOG_UNEXPLORED, dtype=np.int8)
    world = WorldState(
        tick=0,
        elapsed=0.0,
        terrain=stub,
        drones=drones,
        survivors=(),
        fog_grid=fog,
        base_position=base,
    )
    return cw, config, world


def test_simulation_run():
    """Run 200 ticks, print drone/survivor positions, verify detection."""
    cw, config, world = _setup()
    rng = np.random.default_rng(42)

    # ---- Print initial state ----
    print("\n" + "=" * 70)
    print("SIMULATION RUN TEST — 200 ticks")
    print("=" * 70)

    all_survivors = cw.get_all_survivors()
    print("\nWorld: 10240m x 10240m")
    print(f"Base: ({world.base_position.x:.0f}, {world.base_position.z:.0f})")
    print(f"Drones: {len(world.drones)}")
    print(f"Total survivors in world: {len(all_survivors)}")

    print("\n--- SURVIVOR POSITIONS ---")
    for s in all_survivors:
        print(f"  Survivor {s.id:>8d}: ({s.position.x:7.0f}, {s.position.z:7.0f})")

    print("\n--- INITIAL DRONE POSITIONS ---")
    for d in world.drones:
        print(
            f"  Drone {d.id:>2d}: ({d.position.x:7.0f}, {d.position.z:7.0f}) "
            f"status={d.status.name} battery={d.battery:.0f}%"
        )

    # ---- Give drones move commands toward survivor clusters ----
    # Find the nearest survivor to the base
    base = world.base_position
    nearest_survivor = min(
        all_survivors,
        key=lambda s: math.sqrt((s.position.x - base.x) ** 2 + (s.position.z - base.z) ** 2),
    )
    dist_to_nearest = math.sqrt(
        (nearest_survivor.position.x - base.x) ** 2 + (nearest_survivor.position.z - base.z) ** 2,
    )
    print(
        f"\nNearest survivor to base: {nearest_survivor.id} at "
        f"({nearest_survivor.position.x:.0f}, {nearest_survivor.position.z:.0f}), "
        f"distance={dist_to_nearest:.0f}m"
    )

    # Command all drones to spread toward the nearest cluster
    commands = []
    for d in world.drones:
        commands.append(
            Command(
                type="move_to",
                drone_id=d.id,
                target=Vec3(
                    nearest_survivor.position.x + d.id * 50,
                    0,
                    nearest_survivor.position.z + d.id * 50,
                ),
            )
        )

    # ---- Run simulation ----
    total_found = 0
    print("\n--- RUNNING 200 TICKS ---")

    for i in range(200):
        cmds = commands if i == 0 else []
        world = tick_chunked(world, 0.05, cw, cmds, rng=rng, config=config)

        discovered_this_tick = [e for e in world.events if e.type.name == "SURVIVOR_FOUND"]
        if discovered_this_tick:
            for e in discovered_this_tick:
                total_found += 1
                print(
                    f"  [tick {world.tick:>3d}] SURVIVOR FOUND! "
                    f"drone={e.drone_id} survivor={e.survivor_id}"
                )

        if i % 50 == 49 or i == 0:
            active = sum(1 for d in world.drones if d.status == DroneStatus.ACTIVE)
            found = sum(1 for s in world.survivors if s.discovered)
            print(
                f"\n  [tick {world.tick:>3d}] Drones active: {active}/{len(world.drones)} | "
                f"Survivors in range: {len(world.survivors)} | "
                f"Discovered: {found}"
            )
            for d in world.drones[:3]:
                print(
                    f"    Drone {d.id:>2d}: ({d.position.x:7.0f}, {d.position.z:7.0f}) "
                    f"speed={d.velocity.length():.1f} m/s "
                    f"task={d.current_task} battery={d.battery:.0f}%"
                )

    # ---- Final state ----
    print(f"\n{'=' * 70}")
    print(f"FINAL STATE after {world.tick} ticks ({world.elapsed:.0f}s)")
    print(f"{'=' * 70}")

    print("\n--- DRONE FINAL POSITIONS ---")
    for d in world.drones:
        dist_from_base = math.sqrt((d.position.x - base.x) ** 2 + (d.position.z - base.z) ** 2)
        print(
            f"  Drone {d.id:>2d}: ({d.position.x:7.0f}, {d.position.z:7.0f}) "
            f"dist_from_base={dist_from_base:7.0f}m "
            f"status={d.status.name} battery={d.battery:.0f}%"
        )

    found_total = sum(1 for s in world.survivors if s.discovered)
    print(f"\nSurvivors discovered: {found_total}")
    print(f"Survivors in active chunks: {len(world.survivors)}")

    # ---- Assertions ----
    # Drones must have MOVED from the base
    max_dist = max(
        math.sqrt((d.position.x - base.x) ** 2 + (d.position.z - base.z) ** 2) for d in world.drones
    )
    print(f"Max drone distance from base: {max_dist:.0f}m")
    assert max_dist > 100, f"Drones didn't move! Max distance from base: {max_dist:.0f}m"

    # all_survivors must be non-empty
    assert len(all_survivors) > 0, "No survivors in the world at all"

    # all_survivors must be AWAY from base
    survivors_near_base = sum(
        1
        for s in all_survivors
        if math.sqrt((s.position.x - base.x) ** 2 + (s.position.z - base.z) ** 2) < 1000
    )
    survivors_far = len(all_survivors) - survivors_near_base
    print(f"Survivors near base (<1km): {survivors_near_base}")
    print(f"Survivors far from base: {survivors_far}")
    assert survivors_far > 0, "All survivors are at the base"

    print(f"\n{'=' * 70}")
    print("TEST PASSED")
    print(f"{'=' * 70}")


def test_drones_discover_survivors():
    """Run long enough for drones to reach survivors and detect them."""
    cw, config, world = _setup()
    rng = np.random.default_rng(42)
    all_survivors = cw.get_all_survivors()
    base = world.base_position

    print("\n" + "=" * 70)
    print("DISCOVERY TEST — drones fly to survivors and detect them")
    print("=" * 70)

    # Find nearest survivor
    nearest = min(
        all_survivors,
        key=lambda s: math.sqrt((s.position.x - base.x) ** 2 + (s.position.z - base.z) ** 2),
    )
    dist = math.sqrt((nearest.position.x - base.x) ** 2 + (nearest.position.z - base.z) ** 2)
    print(f"Base: ({base.x:.0f}, {base.z:.0f})")
    print(
        f"Nearest survivor: ({nearest.position.x:.0f}, {nearest.position.z:.0f}) dist={dist:.0f}m"
    )

    # Send all drones directly to the nearest survivor
    commands = [
        Command(
            type="move_to",
            drone_id=d.id,
            target=Vec3(
                nearest.position.x + (d.id % 3) * 10, 0, nearest.position.z + (d.id // 3) * 10
            ),
        )
        for d in world.drones
    ]

    # Run enough ticks for drones to arrive (at 15 m/s, dist/15 seconds, at 20Hz)
    ticks_needed = int((dist / 15.0) * 20) + 200  # extra margin
    ticks_needed = min(ticks_needed, 4000)  # cap at 200 seconds
    print(f"Running {ticks_needed} ticks ({ticks_needed / 20:.0f}s sim time)")

    total_found = 0
    for i in range(ticks_needed):
        cmds = commands if i == 0 else []
        world = tick_chunked(world, 0.05, cw, cmds, rng=rng, config=config)
        for e in world.events:
            if e.type.name == "SURVIVOR_FOUND":
                total_found += 1
                # Find the drone and survivor positions
                drone = next((d for d in world.drones if d.id == e.drone_id), None)
                surv = next((s for s in world.survivors if s.id == e.survivor_id), None)
                if drone and surv:
                    d2s = math.sqrt(
                        (drone.position.x - surv.position.x) ** 2
                        + (drone.position.z - surv.position.z) ** 2
                    )
                    print(
                        f"  [tick {world.tick:>4d}] FOUND survivor {e.survivor_id} "
                        f"by drone {e.drone_id} (dist={d2s:.0f}m)"
                    )

        if i % 500 == 499:
            lead = world.drones[0]
            d2target = math.sqrt(
                (lead.position.x - nearest.position.x) ** 2
                + (lead.position.z - nearest.position.z) ** 2
            )
            found = sum(1 for s in world.survivors if s.discovered)
            print(
                f"  [tick {world.tick:>4d}] Drone 0 dist to target: {d2target:.0f}m | "
                f"found: {found}"
            )

    found_total = sum(1 for s in world.survivors if s.discovered)
    print(f"\nTotal survivors discovered: {found_total}")
    print(f"Total discovery events: {total_found}")
    assert found_total > 0, (
        f"No survivors discovered after {ticks_needed} ticks! "
        f"Drones flew to ({nearest.position.x:.0f}, {nearest.position.z:.0f}) "
        f"but detected nothing."
    )
    print(f"\n{'=' * 70}")
    print(f"DISCOVERY TEST PASSED — {found_total} survivors found")
    print(f"{'=' * 70}")
