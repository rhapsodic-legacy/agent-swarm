#!/usr/bin/env python3
"""End-to-end spatial verification for the drone swarm simulator.

Runs the sim for N ticks WITHOUT a server, then checks invariants that
have historically broken across the backend→frontend boundary:

1. Fog clears AT drone positions, not elsewhere
2. Drone positions are near the base at startup
3. Survivor clusters seeded and reachable for the active mission
4. Terrain chunks are generated near the base
5. Fog RLE serialization is spatially correct
6. State message contains all required fields
7. Zone assignments are near the base (not world center)

This catches coordinate-system bugs, UV flips, transposed axes, and
stale-config regressions without needing a browser.

Usage:
    cd backend && uv run python scripts/verify_e2e.py
    cd backend && uv run python scripts/verify_e2e.py lost_hiker
"""

from __future__ import annotations

import json
import math
import os
import sys

# Make `src.*` importable regardless of cwd (script lives in backend/scripts).
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import numpy as np

from src.agents.coordinator import SwarmCoordinator
from src.server.main import _compress_chunk_fog
from src.simulation.engine import tick_chunked
from src.simulation.mission import available_missions, build_mission
from src.simulation.types import (
    FOG_EXPLORED,
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    WorldState,
)
from src.terrain.chunked import ChunkedWorld

# ---------------------------------------------------------------------------
# Config — mirrors main.py
# ---------------------------------------------------------------------------

WORLD_SIZE = 10240
CHUNK_SIZE = 1024
SEED = 42
DEFAULT_MISSION = "aircraft_crash"

sim_config = SimConfig(
    terrain_size=1024,
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

# ---------------------------------------------------------------------------
# Setup — same as main.py simulation_loop
# ---------------------------------------------------------------------------


def setup(mission_name: str = DEFAULT_MISSION):
    """Create world state matching main.py's mission-driven chunked setup."""
    from src.simulation.drone import create_drone_fleet

    mission = build_mission(mission_name, WORLD_SIZE, SEED)
    cw = ChunkedWorld(WORLD_SIZE, CHUNK_SIZE, SEED, sim_config, clusters=mission.clusters)

    base = mission.base_position
    drones = create_drone_fleet(sim_config.drone_count, base, sim_config)

    stub_terrain = Terrain(
        width=WORLD_SIZE,
        height=WORLD_SIZE,
        max_elevation=sim_config.max_elevation,
        heightmap=cw.make_heightmap_proxy(),
        biome_map=cw.make_biome_proxy(),
        survivors=(),
        seed=SEED,
    )

    fog_res = max(WORLD_SIZE // 10, 256)
    global_fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)

    world = WorldState(
        tick=0,
        elapsed=0.0,
        terrain=stub_terrain,
        drones=drones,
        survivors=(),
        fog_grid=global_fog,
        comms_links=(),
        events=(),
        base_position=base,
        tick_rate=sim_config.tick_rate,
    )

    return world, cw, base, fog_res, mission


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        msg = f"{name}: {detail}" if detail else name
        print(f"  {FAIL} {msg}")
        failures.append(msg)


def run_checks(mission_name: str = DEFAULT_MISSION):
    world, cw, base, fog_res, mission = setup(mission_name)
    fog_scale = WORLD_SIZE / fog_res
    dt = 1.0 / sim_config.tick_rate
    rng = np.random.default_rng(SEED)
    coordinator = SwarmCoordinator(sim_config)

    base_x, base_z = base.x, base.z

    print(f"Mission: {mission.title} (base={base_x:.0f},{base_z:.0f})")

    # ------------------------------------------------------------------
    print("\n== 1. Initial drone positions ==")
    # ------------------------------------------------------------------
    for d in world.drones:
        dist = math.sqrt((d.position.x - base_x) ** 2 + (d.position.z - base_z) ** 2)
        if dist > 50:
            check(f"Drone {d.id} near base", False, f"dist={dist:.0f}m from base ({base_x:.0f},{base_z:.0f})")
            break
    else:
        check("All drones start near base", True)

    # Guard against the "base defaulted to exact world center" bug. Missions
    # like disaster_response legitimately stage the base close to center (at
    # the affected-area centroid), so we only rule out exact-center defaults.
    check(
        "Base is not the exact world center default",
        abs(base_x - WORLD_SIZE / 2) > 10 or abs(base_z - WORLD_SIZE / 2) > 10,
        f"base=({base_x:.0f},{base_z:.0f}), center=({WORLD_SIZE/2:.0f},{WORLD_SIZE/2:.0f})",
    )

    # ------------------------------------------------------------------
    print("\n== 2. Run 30 ticks, check fog clears at drone positions ==")
    # ------------------------------------------------------------------
    for i in range(30):
        commands = coordinator.update(world, sim_config)
        world = tick_chunked(world, dt, cw, commands=commands, rng=rng, config=sim_config)

    # Fog should have explored cells
    explored_count = np.count_nonzero(world.fog_grid == FOG_EXPLORED)
    check("Fog has explored cells after 30 ticks", explored_count > 0, f"explored={explored_count}")

    # Check that explored cells are NEAR drone positions, not elsewhere
    explored_coords = np.argwhere(world.fog_grid == FOG_EXPLORED)  # (N, 2) = (row, col)
    if explored_coords.shape[0] > 0:
        # Centroid of explored area in world coords
        mean_row = float(explored_coords[:, 0].mean())
        mean_col = float(explored_coords[:, 1].mean())
        fog_centroid_x = mean_col * fog_scale  # col -> world X
        fog_centroid_z = mean_row * fog_scale  # row -> world Z

        # Centroid of drone positions
        drone_xs = [d.position.x for d in world.drones]
        drone_zs = [d.position.z for d in world.drones]
        drone_centroid_x = sum(drone_xs) / len(drone_xs)
        drone_centroid_z = sum(drone_zs) / len(drone_zs)

        dist = math.sqrt(
            (fog_centroid_x - drone_centroid_x) ** 2
            + (fog_centroid_z - drone_centroid_z) ** 2
        )

        check(
            "Fog clearing centroid matches drone centroid",
            dist < 1500,  # generous — drones spread out
            f"fog_centroid=({fog_centroid_x:.0f},{fog_centroid_z:.0f}), "
            f"drone_centroid=({drone_centroid_x:.0f},{drone_centroid_z:.0f}), "
            f"dist={dist:.0f}m",
        )

        # Fog clearing should be near BASE, not near world center
        dist_to_base = math.sqrt(
            (fog_centroid_x - base_x) ** 2 + (fog_centroid_z - base_z) ** 2
        )
        dist_to_center = math.sqrt(
            (fog_centroid_x - WORLD_SIZE / 2) ** 2
            + (fog_centroid_z - WORLD_SIZE / 2) ** 2
        )
        check(
            "Fog clearing is near base, not world center",
            dist_to_base < dist_to_center,
            f"dist_to_base={dist_to_base:.0f}m, dist_to_center={dist_to_center:.0f}m",
        )

    # ------------------------------------------------------------------
    print("\n== 3. Fog RLE serialization spatial check ==")
    # ------------------------------------------------------------------
    rle_data = _compress_chunk_fog(world.fog_grid)
    rle = rle_data["rle"]
    total_cells = sum(count for _val, count in rle)
    check(
        "RLE cell count matches grid size",
        total_cells == fog_res * fog_res,
        f"rle_total={total_cells}, expected={fog_res * fog_res}",
    )

    # Reconstruct and verify it matches the original
    reconstructed = np.zeros(fog_res * fog_res, dtype=np.int8)
    offset = 0
    for val, count in rle:
        reconstructed[offset : offset + count] = val
        offset += count
    reconstructed = reconstructed.reshape(fog_res, fog_res)
    check("RLE roundtrip matches original fog grid", np.array_equal(reconstructed, world.fog_grid))

    # Find first explored cell in RLE and verify it maps to near-base
    flat_idx = None
    idx = 0
    for val, count in rle:
        if val == FOG_EXPLORED:
            flat_idx = idx
            break
        idx += count

    if flat_idx is not None:
        rle_row = flat_idx // fog_res
        rle_col = flat_idx % fog_res
        rle_world_x = rle_col * fog_scale
        rle_world_z = rle_row * fog_scale
        dist_rle_to_base = math.sqrt(
            (rle_world_x - base_x) ** 2 + (rle_world_z - base_z) ** 2
        )
        check(
            "First explored cell in RLE maps to near base",
            dist_rle_to_base < 3000,
            f"rle_world=({rle_world_x:.0f},{rle_world_z:.0f}), "
            f"base=({base_x:.0f},{base_z:.0f}), dist={dist_rle_to_base:.0f}m",
        )

    # ------------------------------------------------------------------
    # CRITICAL: Simulate frontend texture mapping to verify screen position
    # ------------------------------------------------------------------
    # PlaneGeometry(w,h) after rotateX(-PI/2) with position (w/2, 210, h/2):
    #   UV (0,0) -> world (0, WORLD_SIZE)   i.e. V=0 is Z=max
    #   UV (0,1) -> world (0, 0)            i.e. V=1 is Z=0
    # DataTexture pixel (texX, texY): texY=0 is bottom row -> V=0 -> world Z=max
    # So fog row R at world Z = R*scale needs to go to texY = (fog_res-1-R)
    # Frontend does this flip in updateFromRLE. We verify the math here.
    if flat_idx is not None:
        fog_row = rle_row
        fog_col = rle_col
        # After row-flip in frontend: texRow = fog_res - 1 - fog_row
        tex_row = fog_res - 1 - fog_row
        # texRow maps to V = texRow / fog_res
        v = tex_row / fog_res
        # V=0 -> world Z=WORLD_SIZE, V=1 -> world Z=0
        # screen_z = WORLD_SIZE * (1 - v)
        screen_z = WORLD_SIZE * (1.0 - v)
        screen_x = (fog_col / fog_res) * WORLD_SIZE
        intended_z = fog_row * fog_scale
        intended_x = fog_col * fog_scale

        check(
            "Frontend texture Z matches intended Z (row-flip correct)",
            abs(screen_z - intended_z) < fog_scale * 2,
            f"screen=({screen_x:.0f},{screen_z:.0f}), "
            f"intended=({intended_x:.0f},{intended_z:.0f}), "
            f"Z_error={abs(screen_z - intended_z):.0f}m",
        )

    # ------------------------------------------------------------------
    print("\n== 4. Terrain chunks generated near base ==")
    # ------------------------------------------------------------------
    base_chunks = cw.get_chunks_near(base_x, base_z, 3072.0)
    check("Chunks generated near base", len(base_chunks) > 0, f"count={len(base_chunks)}")

    # All chunk origins should be within reasonable distance of base
    for chunk in base_chunks:
        ox = chunk.coord.cx * CHUNK_SIZE + CHUNK_SIZE / 2
        oz = chunk.coord.cz * CHUNK_SIZE + CHUNK_SIZE / 2
        dist = math.sqrt((ox - base_x) ** 2 + (oz - base_z) ** 2)
        if dist > 4500:
            check(
                f"Chunk ({chunk.coord.cx},{chunk.coord.cz}) near base",
                False,
                f"chunk_center=({ox:.0f},{oz:.0f}), dist={dist:.0f}m",
            )
            break
    else:
        check("All pre-generated chunks are near base", True)

    # ------------------------------------------------------------------
    print("\n== 5. Mission clusters seeded and reachable ==")
    # ------------------------------------------------------------------
    # Post-Phase-2, missions stage the base *near* their high-mass cluster
    # (see memory/project_reach_bottleneck.md). The old "cluster far from base"
    # invariant is gone — the new one is "at least one cluster is within
    # fresh-drone round-trip reach, and generated chunks at that cluster
    # contain survivors."
    from src.terrain.chunked import ChunkCoord

    clusters = cw._survivor_clusters
    check(
        "Mission seeded at least one cluster",
        len(clusters) > 0,
        f"cluster_count={len(clusters)}",
    )

    # Verify the highest-weight cluster is within practical reach of the base.
    # 3km is the conservative bound: fresh-drone reach is ~950m radius but
    # missions may stage clusters up to ~2.5km when the prior mass is spread.
    if clusters:
        cluster_dists = [
            (w, math.sqrt((cx - base_x) ** 2 + (cz - base_z) ** 2), cx, cz)
            for (cx, cz, _r, w) in clusters
        ]
        cluster_dists.sort(key=lambda t: -t[0])  # by weight desc
        _top_w, top_dist, top_x, top_z = cluster_dists[0]
        check(
            "Highest-weight cluster is reachable from base",
            top_dist < 3000.0,
            f"top_cluster=({top_x:.0f},{top_z:.0f}), base=({base_x:.0f},{base_z:.0f}), dist={top_dist:.0f}m",
        )

        # Generate the chunk containing the top cluster — survivors should
        # spawn there.
        cluster_chunk_cx = int(top_x // CHUNK_SIZE)
        cluster_chunk_cz = int(top_z // CHUNK_SIZE)
        cluster_chunk = cw.get_chunk(ChunkCoord(cluster_chunk_cx, cluster_chunk_cz))
        check(
            "Survivors placed near top cluster",
            len(cluster_chunk.survivors) > 0,
            f"chunk ({cluster_chunk_cx},{cluster_chunk_cz}) has "
            f"{len(cluster_chunk.survivors)} survivors",
        )

    # Mission metadata sanity: briefing dict has all fields the frontend
    # IntelBriefing overlay expects.
    briefing = mission.to_briefing_dict()
    required_keys = {
        "name", "title", "description", "known_facts",
        "base_position", "survival_window_seconds", "intel_pins",
    }
    missing = required_keys - set(briefing.keys())
    check(
        "Mission briefing has all required fields",
        not missing,
        f"missing={missing}",
    )
    check(
        "Mission briefing has >=1 intel pin",
        len(briefing["intel_pins"]) > 0,
        f"pin_count={len(briefing['intel_pins'])}",
    )

    # ------------------------------------------------------------------
    print("\n== 6. Zone assignments near base ==")
    # ------------------------------------------------------------------
    for drone_id, agent in coordinator.agents.items():
        if agent.zone is not None:
            (r_min, c_min), (r_max, c_max) = agent.zone
            zone_center_x = (c_min + c_max) / 2.0
            zone_center_z = (r_min + r_max) / 2.0
            dist = math.sqrt((zone_center_x - base_x) ** 2 + (zone_center_z - base_z) ** 2)
            if dist > 4000:
                check(
                    f"Drone {drone_id} zone near base",
                    False,
                    f"zone_center=({zone_center_x:.0f},{zone_center_z:.0f}), dist={dist:.0f}m",
                )
                break
    else:
        check("All zone assignments are near base", True)

    # ------------------------------------------------------------------
    print("\n== 7. Frontend data contract ==")
    # ------------------------------------------------------------------
    # Simulate what main.py sends — verify required fields
    chunk_data = cw.serialize_chunk(base_chunks[0].coord)
    chunk_json = json.dumps(chunk_data)
    check("Chunk has resolution field", "resolution" in chunk_data)
    check(
        "Chunk resolution <= 256",
        chunk_data.get("resolution", 9999) <= 256,
        f"resolution={chunk_data.get('resolution')}",
    )
    check(
        "Chunk JSON < 500KB",
        len(chunk_json) < 500_000,
        f"size={len(chunk_json)} bytes",
    )

    # Activity log
    log = coordinator.get_recent_log()
    check("Activity log has entries after 30 ticks", len(log) > 0, f"count={len(log)}")

    # ------------------------------------------------------------------
    print("\n== 8. Fog texture coordinate sanity ==")
    # ------------------------------------------------------------------
    # The fog grid cell at (row=R, col=C) should map to world (C*scale, R*scale).
    # On the frontend, the DataTexture pixel at (texX, texY) maps to:
    #   - PlaneGeometry UV (u, v) where u = texX/width, v = texY/height
    #   - After rotateX(-PI/2), geometry local (x, y) -> world (x, z)
    #   - PlaneGeometry center at (worldWidth/2, worldHeight/2)
    #   - UV (0,0) -> world (0, 0) if flipY=false
    #   - UV (0,0) -> world (0, worldHeight) if flipY=true  <-- THIS IS THE BUG
    #
    # The fog grid row 0 = Z=0 (south/bottom). Without flipY, texture row 0
    # maps to UV v=0 which is the bottom of the plane -> world Z=0. CORRECT.
    # With flipY=true (Three.js default), texture row 0 maps to UV v=1
    # which is the top of the plane -> world Z=worldHeight. WRONG.
    #
    # Verify: explored cells near row=150 (world Z=1500) should NOT appear
    # near row=874 (world Z=8740) which is where flipY would put them.
    explored_rows = explored_coords[:, 0] if explored_coords.shape[0] > 0 else np.array([])
    if len(explored_rows) > 0:
        mean_explored_row = float(explored_rows.mean())
        expected_row = base_z / fog_scale  # ~153
        flipped_row = fog_res - expected_row  # ~871

        check(
            "Fog explored rows are near base (not flipped)",
            abs(mean_explored_row - expected_row) < abs(mean_explored_row - flipped_row),
            f"mean_explored_row={mean_explored_row:.0f}, "
            f"expected={expected_row:.0f}, flipped_would_be={flipped_row:.0f}",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures:
        print(f"\033[91m{len(failures)} FAILED:\033[0m")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print(f"\033[92mAll checks passed.\033[0m")
        return 0


if __name__ == "__main__":
    # Default: verify the aircraft_crash mission (the historical default).
    # Pass "all" to run against every registered mission, or a specific name.
    arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MISSION
    if arg == "all":
        exit_code = 0
        for name in available_missions():
            print(f"\n{'=' * 60}\n  MISSION: {name}\n{'=' * 60}")
            failures.clear()
            rc = run_checks(name)
            exit_code = max(exit_code, rc)
        sys.exit(exit_code)
    sys.exit(run_checks(arg))
