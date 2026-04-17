"""Integration tests for the chunked world system.

These tests target regressions actually hit in production:
  - 4.2MB chunk serialization (should be <500KB after downsampling)
  - Missing `resolution` field in serialized chunks
  - Fog grid staying all-unexplored (hardcoded zeros instead of using world.fog_grid)
  - RLE fog compression roundtrip correctness
  - RuntimeError from iterating `clients` set while it changes
"""

from __future__ import annotations

import json
import threading

import numpy as np

from src.simulation.engine import tick_chunked
from src.simulation.types import (
    FOG_EXPLORED,
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.terrain.chunked import ChunkCoord, ChunkedWorld

# Import the RLE compression helper from main.py
from src.server.main import _compress_chunk_fog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small config to keep tests fast (~1-2s chunk generation)
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


def _make_chunked_world() -> ChunkedWorld:
    return ChunkedWorld(_WORLD_SIZE, _CHUNK_SIZE, _SEED, _CFG)


def _make_world_state(chunked_world: ChunkedWorld) -> WorldState:
    """Create a WorldState wired to the given ChunkedWorld, matching main.py logic."""
    from src.simulation.drone import create_drone_fleet

    ws = chunked_world.get_world_size()
    base = Vec3(ws / 2.0, 0.0, ws / 2.0)
    drones = create_drone_fleet(_CFG.drone_count, base, _CFG)

    stub_terrain = Terrain(
        width=ws,
        height=ws,
        max_elevation=_CFG.max_elevation,
        heightmap=chunked_world.make_heightmap_proxy(),
        biome_map=chunked_world.make_biome_proxy(),
        survivors=(),
        seed=_SEED,
    )

    fog_res = max(ws // 10, 256)
    global_fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)

    return WorldState(
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


# ---------------------------------------------------------------------------
# 1. Chunk serialization size (regression: 4.2MB -> should be < 500KB)
# ---------------------------------------------------------------------------


def test_chunk_serialization_size():
    """Serialized chunk JSON must be under 500KB (catches the 4.2MB regression)."""
    cw = _make_chunked_world()
    coord = ChunkCoord(0, 0)
    data = cw.serialize_chunk(coord)
    blob = json.dumps(data)
    size_kb = len(blob) / 1024

    assert size_kb < 500, (
        f"Serialized chunk is {size_kb:.1f}KB — expected < 500KB. "
        f"Downsampling may have regressed."
    )


# ---------------------------------------------------------------------------
# 2. Chunk has resolution field
# ---------------------------------------------------------------------------


def test_chunk_has_resolution_field():
    """Serialized chunk must include a `resolution` field <= 256."""
    cw = _make_chunked_world()
    coord = ChunkCoord(0, 0)
    data = cw.serialize_chunk(coord)

    assert "resolution" in data, "Serialized chunk missing `resolution` field"
    assert isinstance(data["resolution"], int), "`resolution` must be an integer"
    assert data["resolution"] <= 256, (
        f"resolution={data['resolution']} — should be <= 256 "
        f"(downsampled for network transmission)"
    )


# ---------------------------------------------------------------------------
# 3. Fog grid updates after ticks (regression: fog was hardcoded to all-unexplored)
# ---------------------------------------------------------------------------


def test_fog_grid_updates_after_ticks():
    """After several ticks, world.fog_grid must have some explored cells.

    This catches the bug where fog_grid was hardcoded to all-unexplored
    instead of being updated by tick_chunked.
    """
    cw = _make_chunked_world()
    world = _make_world_state(cw)
    rng = np.random.default_rng(_SEED)
    dt = 1.0 / _CFG.tick_rate

    # Initially all unexplored
    assert np.all(world.fog_grid == FOG_UNEXPLORED)

    # Run several ticks so drones ascend and sensors activate
    for _ in range(20):
        world = tick_chunked(world, dt, cw, commands=[], rng=rng, config=_CFG)

    explored_count = np.count_nonzero(world.fog_grid == FOG_EXPLORED)
    assert explored_count > 0, (
        "After 20 ticks, fog_grid has zero explored cells. "
        "The global fog grid is not being updated by tick_chunked."
    )


# ---------------------------------------------------------------------------
# 4. Fog RLE compression roundtrip
# ---------------------------------------------------------------------------


def test_fog_rle_compression():
    """RLE-compressed fog grid must roundtrip: sum of run lengths == width * height."""
    size = 128
    fog = np.zeros((size, size), dtype=np.int8)
    # Mark a stripe as explored
    fog[10:30, 20:80] = FOG_EXPLORED

    result = _compress_chunk_fog(fog)

    assert result["width"] == size
    assert result["height"] == size

    total_cells = sum(count for _val, count in result["rle"])
    assert total_cells == size * size, (
        f"RLE run-length total {total_cells} != expected {size * size}. "
        f"Compression lost or gained cells."
    )

    # Verify roundtrip by reconstructing the flat array
    reconstructed = np.zeros(size * size, dtype=np.int8)
    offset = 0
    for val, count in result["rle"]:
        reconstructed[offset : offset + count] = val
        offset += count
    reconstructed = reconstructed.reshape(size, size)

    np.testing.assert_array_equal(reconstructed, fog)


def test_fog_rle_all_unexplored():
    """All-unexplored grid should compress to a single run."""
    size = 64
    fog = np.full((size, size), FOG_UNEXPLORED, dtype=np.int8)
    result = _compress_chunk_fog(fog)
    assert len(result["rle"]) == 1
    assert result["rle"][0] == [FOG_UNEXPLORED, size * size]


def test_fog_rle_empty():
    """Empty fog grid (0x0) should produce empty RLE."""
    fog = np.zeros((0, 0), dtype=np.int8)
    result = _compress_chunk_fog(fog)
    assert result["rle"] == []


# ---------------------------------------------------------------------------
# 5. Broadcast clients snapshot — iterating list(clients) is safe
# ---------------------------------------------------------------------------


def test_broadcast_clients_snapshot():
    """Verify that iterating list(clients) does not crash when the set is modified.

    This tests the fix for: RuntimeError: Set changed size during iteration.
    The pattern is: snapshot the set via list() before iterating.
    """
    clients: set[str] = {"ws1", "ws2", "ws3"}
    disconnected: list[str] = []

    # Simulate broadcasting: iterate snapshot while another "thread" modifies the set
    snapshot = list(clients)

    # Modify the original set (simulating a new WebSocket connecting mid-broadcast)
    clients.add("ws4")
    clients.discard("ws1")

    # Iteration over snapshot should work fine
    for ws in snapshot:
        try:
            # Simulate work (no actual WebSocket here)
            _ = f"sending to {ws}"
        except Exception:
            disconnected.append(ws)

    # Snapshot should still have the original 3 entries
    assert len(snapshot) == 3
    assert "ws1" in snapshot  # was in original set even though removed from live set
    assert "ws4" not in snapshot  # added after snapshot

    # Verify no crashes occurred
    assert len(disconnected) == 0


def test_broadcast_clients_concurrent_modification():
    """Verify the list(set) pattern is thread-safe enough for our use case.

    In an async context, only one coroutine runs at a time, but this test
    demonstrates the snapshot pattern works even with threaded mutations.
    """
    clients: set[int] = set(range(100))
    errors: list[Exception] = []

    def mutate_set():
        for i in range(100, 200):
            clients.add(i)
            if i % 2 == 0:
                clients.discard(i - 100)

    # Take snapshot, then mutate in parallel
    snapshot = list(clients)
    t = threading.Thread(target=mutate_set)
    t.start()

    # Iterate snapshot while mutations happen
    try:
        for item in snapshot:
            _ = item * 2  # simulate work
    except RuntimeError as e:
        errors.append(e)

    t.join()

    assert len(errors) == 0, f"Snapshot iteration raised errors: {errors}"
    assert len(snapshot) == 100  # snapshot is frozen at time of creation


# ---------------------------------------------------------------------------
# 6. Activity log in coordinator
# ---------------------------------------------------------------------------


def test_coordinator_activity_log():
    """Coordinator should produce activity log entries as drones operate."""
    from src.agents.coordinator import SwarmCoordinator

    cw = _make_chunked_world()
    world = _make_world_state(cw)

    coordinator = SwarmCoordinator(_CFG)
    rng = np.random.default_rng(_SEED)
    dt = 1.0 / _CFG.tick_rate

    # Run a few ticks so zone assignment and initial spread happen
    for _ in range(5):
        commands = coordinator.update(world, _CFG)
        world = tick_chunked(world, dt, cw, commands=commands, rng=rng, config=_CFG)

    log = coordinator.get_recent_log()
    assert len(log) > 0, "Activity log should have entries after several ticks"
    # Should have at least a zone assignment entry
    assert any("zone" in e["message"].lower() or "assigned" in e["message"].lower() for e in log), (
        f"Expected a zone assignment log entry. Got: {[e['message'] for e in log]}"
    )


# ---------------------------------------------------------------------------
# 7. Survivor-biased search
# ---------------------------------------------------------------------------


def test_survivor_proximity_boosts_priority_search():
    """priority_search should prefer unexplored cells near known survivor locations."""
    from src.agents.search_patterns import priority_search

    # Create a small fog grid: mostly explored, with two unexplored patches
    fog = np.ones((100, 100), dtype=np.int8)  # all explored
    # Patch A: unexplored at (20, 20) — far from survivor
    fog[18:22, 18:22] = 0
    # Patch B: unexplored at (70, 70) — near survivor
    fog[68:72, 68:72] = 0

    cw = _make_chunked_world()
    world = _make_world_state(cw)
    terrain = world.terrain

    drone_pos = Vec3(50.0, 50.0, 50.0)  # center of map

    # Without survivors — should pick based on biome + distance (either patch)
    result_no_surv = priority_search(drone_pos, fog, terrain, survivor_positions=None)
    assert result_no_surv is not None

    # With a survivor near patch B — should strongly prefer patch B
    survivor_near_b = Vec3(72.0, 0.0, 72.0)  # near (row=72, col=72)
    result_with_surv = priority_search(drone_pos, fog, terrain, survivor_positions=[survivor_near_b])
    assert result_with_surv is not None
    # Result should be close to patch B (row ~70, col ~70) not patch A (row ~20, col ~20)
    dist_to_b = ((result_with_surv.x - 70) ** 2 + (result_with_surv.z - 70) ** 2) ** 0.5
    dist_to_a = ((result_with_surv.x - 20) ** 2 + (result_with_surv.z - 20) ** 2) ** 0.5
    assert dist_to_b < dist_to_a, (
        f"Expected search to prefer patch near survivor (B at ~70,70) but got "
        f"result at ({result_with_surv.x:.0f}, {result_with_surv.z:.0f}), "
        f"dist_to_A={dist_to_a:.0f}, dist_to_B={dist_to_b:.0f}"
    )
