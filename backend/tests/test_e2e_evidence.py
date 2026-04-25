"""End-to-end test: evidence discovery flows through the tick pipeline.

Plants evidence in a world, positions a drone right on top of it, ticks
the sim, and verifies:
  1. The drone discovers the evidence.
  2. An EVIDENCE_FOUND SimEvent is emitted with the right payload.
  3. The PoC posterior is reshaped (hottest cell shifts toward the clue).
  4. The evidence tuple in WorldState is updated (discovered=True).

This is the load-bearing integration test for Phase 3 — if this passes,
the whole pipeline from planting → detection → posterior update → event
broadcast → state serialization works.
"""

from __future__ import annotations

import numpy as np

from src.agents.coordinator import SwarmCoordinator
from src.simulation.engine import tick_chunked
from src.simulation.mission import build_mission
from src.simulation.search_map import SearchMap
from src.simulation.types import (
    FOG_UNEXPLORED,
    Drone,
    DroneStatus,
    EventType,
    Evidence,
    EvidenceKind,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.terrain.chunked import ChunkedWorld

WORLD_SIZE = 10240
CHUNK_SIZE = 1024
SEED = 42


def _setup_world_with_evidence(
    evidence_list: list[Evidence],
    drone_pos: Vec3,
) -> tuple[WorldState, ChunkedWorld, SimConfig]:
    """Build a minimal WorldState wrapping the provided evidence + drone."""
    config = SimConfig(
        terrain_size=CHUNK_SIZE,
        drone_count=1,
        survivor_count=5,
        drone_sensor_range=40.0,
        drone_comms_range=120.0,
        drone_battery_drain_rate=0.05,
    )
    # Use a trivial cluster layout so the ChunkedWorld is happy
    cw = ChunkedWorld(
        WORLD_SIZE,
        CHUNK_SIZE,
        SEED,
        config,
        clusters=[(5000.0, 5000.0, 300.0, 1.0)],
    )
    drone = Drone(
        id=0,
        position=drone_pos,
        status=DroneStatus.ACTIVE,
        sensor_active=True,
        sensor_range=40.0,
        battery=100.0,
    )
    stub = Terrain(
        width=WORLD_SIZE,
        height=WORLD_SIZE,
        max_elevation=200.0,
        heightmap=cw.make_heightmap_proxy(),
        biome_map=cw.make_biome_proxy(),
        survivors=(),
        seed=SEED,
    )
    fog_res = max(WORLD_SIZE // 10, 256)
    fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)
    sm = SearchMap.empty(world_size=float(WORLD_SIZE), cell_size=40.0)
    # Seed a weak uniform prior so we have a starting distribution.
    sm.poc[:, :] = np.float32(1.0 / sm.poc.size)
    sm.normalize()
    world = WorldState(
        tick=0,
        elapsed=0.0,
        terrain=stub,
        drones=(drone,),
        survivors=(),
        fog_grid=fog,
        base_position=Vec3(1500.0, 0.0, 1500.0),
        search_map=sm,
        evidence=tuple(evidence_list),
    )
    return world, cw, config


def test_drone_on_top_of_signal_fire_discovers_it() -> None:
    ev = Evidence(
        id=7,
        position=Vec3(5000.0, 0.0, 5000.0),
        kind=EvidenceKind.SIGNAL_FIRE.value,
        confidence=0.9,
    )
    world, cw, config = _setup_world_with_evidence(
        [ev],
        Vec3(5000.0, 50.0, 5000.0),
    )

    # Take one tick
    new_world = tick_chunked(world, dt=0.1, chunked_world=cw, config=config)

    # Evidence should now be marked discovered
    found = [e for e in new_world.evidence if e.discovered]
    assert len(found) == 1
    assert found[0].id == 7
    assert found[0].discovered_by == 0

    # Event emitted
    event_types = [e.type for e in new_world.events]
    assert EventType.EVIDENCE_FOUND in event_types


def test_evidence_discovery_reshapes_poc_posterior() -> None:
    """Evidence discovery inside the tick pipeline should reshape the PoC
    posterior toward the clue's neighborhood.

    Using uniform prior here so the multiplicative boost can actually
    concentrate mass near the clue (see test_signal_fire_reshapes_mass_toward_clue
    in test_evidence.py for the nuance)."""
    world, cw, config = _setup_world_with_evidence(
        [
            Evidence(
                id=1,
                position=Vec3(7500.0, 0.0, 7500.0),
                kind=EvidenceKind.SIGNAL_FIRE.value,
                confidence=0.9,
            )
        ],
        Vec3(7500.0, 50.0, 7500.0),
    )
    sm = world.search_map  # type: ignore[assignment]

    def mass_near(cx: float, cz: float, radius: float) -> float:
        col, row = sm.world_to_cell(cx, cz)
        rc = int(radius / sm.cell_size)
        r_min = max(0, row - rc)
        r_max = min(sm.poc.shape[0], row + rc + 1)
        c_min = max(0, col - rc)
        c_max = min(sm.poc.shape[1], col + rc + 1)
        return float(sm.poc[r_min:r_max, c_min:c_max].sum())

    before_mass = mass_near(7500, 7500, 600.0)

    new_world = tick_chunked(world, dt=0.1, chunked_world=cw, config=config)

    after_mass = mass_near(7500, 7500, 600.0)
    assert after_mass > before_mass * 1.5

    # And the evidence is marked discovered
    assert any(e.discovered for e in new_world.evidence)


def test_evidence_far_from_drone_stays_undiscovered() -> None:
    """A clue outside detection range is not picked up."""
    ev = Evidence(
        id=1,
        position=Vec3(5000.0, 0.0, 5000.0),
        kind=EvidenceKind.FOOTPRINT.value,
        confidence=0.6,
    )
    # Drone 500m away — way outside the 60m footprint base range.
    world, cw, config = _setup_world_with_evidence(
        [ev],
        Vec3(5500.0, 50.0, 5500.0),
    )
    new_world = tick_chunked(world, dt=0.1, chunked_world=cw, config=config)
    assert all(not e.discovered for e in new_world.evidence)
    assert not any(e.type == EventType.EVIDENCE_FOUND for e in new_world.events)


def test_lost_hiker_mission_evidence_flows_into_world_state() -> None:
    """Building a mission, then initializing WorldState with the mission's
    evidence, gives the simulation access to it."""
    m = build_mission("lost_hiker", WORLD_SIZE, seed=1)
    assert len(m.evidence) > 0

    config = SimConfig(drone_count=1, survivor_count=1)
    cw = ChunkedWorld(WORLD_SIZE, CHUNK_SIZE, SEED, config, clusters=m.clusters)
    world = WorldState(
        tick=0,
        elapsed=0.0,
        terrain=Terrain(
            width=WORLD_SIZE,
            height=WORLD_SIZE,
            max_elevation=200.0,
            heightmap=cw.make_heightmap_proxy(),
            biome_map=cw.make_biome_proxy(),
            survivors=(),
            seed=SEED,
        ),
        drones=(),
        survivors=(),
        fog_grid=np.zeros((256, 256), dtype=np.int8),
        base_position=m.base_position,
        evidence=tuple(m.evidence),
    )
    # Pre-tick: all evidence undiscovered.
    assert len(world.evidence) == len(m.evidence)
    assert all(not e.discovered for e in world.evidence)


def test_coordinator_logs_evidence_discovery() -> None:
    """The coordinator should append an 'event' entry to its activity log
    when the engine produces an EVIDENCE_FOUND event."""
    ev = Evidence(
        id=3,
        position=Vec3(5000.0, 0.0, 5000.0),
        kind=EvidenceKind.SIGNAL_FIRE.value,
        confidence=0.9,
    )
    world, cw, config = _setup_world_with_evidence(
        [ev],
        Vec3(5000.0, 50.0, 5000.0),
    )
    new_world = tick_chunked(world, dt=0.1, chunked_world=cw, config=config)

    coord = SwarmCoordinator(config)
    coord.update(new_world, config)

    messages = [entry["message"] for entry in coord.get_recent_log()]
    assert any("Evidence" in m for m in messages)
