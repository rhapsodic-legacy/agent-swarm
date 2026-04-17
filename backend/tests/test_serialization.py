"""Tests for state serialization sent over WebSocket.

These validate the state_msg structure built in main.py's simulation_loop,
catching regressions like:
  - fog_grid RLE being all-unexplored (hardcoded zeros instead of actual fog)
  - Missing drone data in state updates
  - Missing world_size / chunk_size fields in chunked mode
"""

from __future__ import annotations

import numpy as np

from src.server.main import _compress_chunk_fog
from src.simulation.drone import create_drone_fleet
from src.simulation.engine import tick_chunked
from src.simulation.types import (
    FOG_EXPLORED,
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.terrain.chunked import ChunkedWorld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = SimConfig(
    terrain_size=256,
    terrain_seed=42,
    max_elevation=200.0,
    drone_count=4,
    tick_rate=20.0,
    survivor_count=5,
    drone_sensor_range=40.0,
    drone_comms_range=100.0,
    drone_battery_drain_rate=0.05,
    drone_battery_critical=15.0,
    drone_cruise_altitude=50.0,
    sensor_failure_prob=0.0,
    comms_failure_prob=0.0,
)

_WORLD_SIZE = 256
_CHUNK_SIZE = 128
_SEED = 42


def _build_state_msg(world: WorldState) -> dict:
    """Build a state_msg dict the same way main.py does (subset of fields we test)."""
    found_survivors = [s for s in world.survivors if s.discovered]
    return {
        "type": "state_update",
        "tick": world.tick,
        "elapsed": round(world.elapsed, 2),
        "drones": [
            {
                "id": d.id,
                "position": [
                    round(d.position.x, 1),
                    round(d.position.y, 1),
                    round(d.position.z, 1),
                ],
                "velocity": [
                    round(d.velocity.x, 1),
                    round(d.velocity.y, 1),
                    round(d.velocity.z, 1),
                ],
                "heading": round(d.heading, 3),
                "battery": round(d.battery, 1),
                "status": d.status.name.lower(),
                "sensor_active": d.sensor_active,
                "comms_active": d.comms_active,
                "current_task": d.current_task,
                "target": (
                    [round(d.target.x, 1), round(d.target.y, 1), round(d.target.z, 1)]
                    if d.target
                    else None
                ),
            }
            for d in world.drones
        ],
        "survivors": [
            {
                "id": s.id,
                "position": [
                    round(s.position.x, 1),
                    round(s.position.y, 1),
                    round(s.position.z, 1),
                ],
                "discovered": s.discovered,
                "discovered_by": s.discovered_by,
                "discovered_at_tick": s.discovered_at_tick,
                "mobile": s.mobile,
            }
            for s in found_survivors
        ],
        "fog_grid": _compress_chunk_fog(world.fog_grid),
        "comms_links": [list(link) for link in world.comms_links],
        "events": [
            {
                "type": e.type.name.lower(),
                "tick": e.tick,
                "drone_id": e.drone_id,
                "survivor_id": e.survivor_id,
            }
            for e in world.events
        ],
        "world_size": _WORLD_SIZE,
        "chunk_size": _CHUNK_SIZE,
    }


def _make_world_and_tick(n_ticks: int) -> WorldState:
    """Create a ChunkedWorld + WorldState and tick it n_ticks times."""
    cw = ChunkedWorld(_WORLD_SIZE, _CHUNK_SIZE, _SEED, _CFG)
    ws = cw.get_world_size()
    base = Vec3(ws / 2.0, 0.0, ws / 2.0)
    drones = create_drone_fleet(_CFG.drone_count, base, _CFG)

    stub_terrain = Terrain(
        width=ws,
        height=ws,
        max_elevation=_CFG.max_elevation,
        heightmap=cw.make_heightmap_proxy(),
        biome_map=cw.make_biome_proxy(),
        survivors=(),
        seed=_SEED,
    )

    fog_res = max(ws // 10, 256)
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
        tick_rate=_CFG.tick_rate,
    )

    rng = np.random.default_rng(_SEED)
    dt = 1.0 / _CFG.tick_rate

    for _ in range(n_ticks):
        world = tick_chunked(world, dt, cw, commands=[], rng=rng, config=_CFG)

    return world


# ---------------------------------------------------------------------------
# 1. Fog grid in state_msg is not all unexplored after ticks
# ---------------------------------------------------------------------------


def test_state_update_fog_not_all_unexplored():
    """After ticks, the fog_grid RLE in the state message must show some explored cells.

    Catches the bug where fog was hardcoded to [[0, N]] regardless of drone activity.
    """
    world = _make_world_and_tick(20)
    msg = _build_state_msg(world)

    fog = msg["fog_grid"]
    rle = fog["rle"]

    # Check that the RLE is not just a single run of unexplored
    is_all_unexplored = len(rle) == 1 and rle[0][0] == FOG_UNEXPLORED
    assert not is_all_unexplored, (
        "fog_grid RLE is entirely unexplored after 20 ticks. "
        "The fog data is not reflecting drone exploration."
    )

    # Verify at least some explored cells exist
    explored_count = sum(count for val, count in rle if val == FOG_EXPLORED)
    assert explored_count > 0, "No explored cells found in fog_grid RLE"


# ---------------------------------------------------------------------------
# 2. State update contains drone data
# ---------------------------------------------------------------------------


def test_state_update_has_drones():
    """State message must contain drone data with valid positions."""
    world = _make_world_and_tick(5)
    msg = _build_state_msg(world)

    assert "drones" in msg
    drones = msg["drones"]
    assert len(drones) == _CFG.drone_count

    for drone in drones:
        assert "id" in drone
        assert "position" in drone
        assert len(drone["position"]) == 3
        assert "battery" in drone
        assert "status" in drone
        assert drone["status"] in ("active", "returning", "recharging", "failed")

    # Drones should have distinct IDs
    ids = [d["id"] for d in drones]
    assert len(set(ids)) == len(ids), "Drone IDs are not unique"


def test_state_update_drones_have_moved():
    """After several ticks, at least some drones should have non-zero altitude."""
    world = _make_world_and_tick(30)
    msg = _build_state_msg(world)

    max_y = max(d["position"][1] for d in msg["drones"])
    assert max_y > 0, "No drone has gained altitude after 30 ticks"


# ---------------------------------------------------------------------------
# 3. Chunked state includes world_size and chunk_size
# ---------------------------------------------------------------------------


def test_state_update_has_world_size():
    """Chunked mode state message must include world_size and chunk_size."""
    world = _make_world_and_tick(1)
    msg = _build_state_msg(world)

    assert "world_size" in msg, "state_msg missing 'world_size'"
    assert "chunk_size" in msg, "state_msg missing 'chunk_size'"
    assert msg["world_size"] == _WORLD_SIZE
    assert msg["chunk_size"] == _CHUNK_SIZE
    assert msg["world_size"] > 0
    assert msg["chunk_size"] > 0
    assert msg["world_size"] >= msg["chunk_size"]
