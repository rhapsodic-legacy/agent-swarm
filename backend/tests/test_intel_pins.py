"""Tests for intel pins (LLM- / operator-injected point priorities)."""

from __future__ import annotations

import numpy as np

from src.agents.coordinator import IntelPin, SwarmCoordinator
from src.simulation.search_map import SearchMap
from src.simulation.types import (
    Command,
    Drone,
    DroneStatus,
    FOG_UNEXPLORED,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)


def _minimal_world(base: Vec3 = Vec3(1000.0, 0.0, 1000.0), tick: int = 0) -> WorldState:
    """Tiny world state sufficient for coordinator.apply_* tests."""
    size = 2048
    heightmap = np.zeros((size, size), dtype=np.float32)
    biome_map = np.zeros((size, size), dtype=np.int8)
    terrain = Terrain(
        width=size, height=size, max_elevation=100.0,
        heightmap=heightmap, biome_map=biome_map, survivors=(), seed=0,
    )
    drone = Drone(id=0, position=base, battery=100.0, status=DroneStatus.ACTIVE)
    fog = np.full((64, 64), FOG_UNEXPLORED, dtype=np.int8)
    return WorldState(
        tick=tick, elapsed=tick * 0.1, terrain=terrain,
        drones=(drone,), survivors=(), fog_grid=fog, comms_links=(), events=(),
        base_position=base, tick_rate=10.0, search_map=None,
    )


def _coord() -> SwarmCoordinator:
    return SwarmCoordinator(SimConfig(drone_battery_drain_rate=0.02))


# ---------------------------------------------------------------- lifecycle

def test_apply_create_stores_pin() -> None:
    coord = _coord()
    world = _minimal_world()
    cmd = Command(
        type="set_intel_pin",
        zone_id="p1",
        data={
            "action": "create",
            "pin_id": "p1",
            "position": [1500.0, 1500.0],
            "radius": 500.0,
            "value": 1.5,
            "label": "last known sighting",
        },
    )
    coord.apply_intel_pin_command(cmd, world)
    assert "p1" in coord.intel_pins
    pin = coord.intel_pins["p1"]
    assert pin.x == 1500.0 and pin.z == 1500.0
    assert pin.radius == 500.0
    assert pin.value == 1.5
    assert pin.label == "last known sighting"
    assert pin.expires_tick is None


def test_apply_ttl_sets_expires_tick() -> None:
    coord = _coord()
    world = _minimal_world(tick=50)
    cmd = Command(
        type="set_intel_pin",
        zone_id="tt",
        data={
            "action": "create", "pin_id": "tt",
            "position": [100.0, 100.0], "ttl_s": 30.0,
        },
    )
    coord.apply_intel_pin_command(cmd, world)
    pin = coord.intel_pins["tt"]
    # tick_rate 10Hz × 30s = 300 ticks, created at tick 50 → expires at 350
    assert pin.expires_tick == 350


def test_apply_delete_removes_pin() -> None:
    coord = _coord()
    world = _minimal_world()
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id="gone", data={
            "action": "create", "pin_id": "gone", "position": [500.0, 500.0],
        }),
        world,
    )
    assert "gone" in coord.intel_pins
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id="gone", data={
            "action": "delete", "pin_id": "gone",
        }),
        world,
    )
    assert "gone" not in coord.intel_pins


def test_apply_clear_removes_all_pins() -> None:
    coord = _coord()
    world = _minimal_world()
    for i in range(3):
        coord.apply_intel_pin_command(
            Command(type="set_intel_pin", zone_id=f"p{i}", data={
                "action": "create", "pin_id": f"p{i}",
                "position": [100.0 * i, 0.0],
            }),
            world,
        )
    assert len(coord.intel_pins) == 3
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id=None, data={"action": "clear"}),
        world,
    )
    assert coord.intel_pins == {}


def test_expire_tick_drops_pin() -> None:
    coord = _coord()
    world = _minimal_world(tick=0)
    # TTL 5 ticks at 10Hz = 0.5s
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id="transient", data={
            "action": "create", "pin_id": "transient",
            "position": [100.0, 100.0], "ttl_s": 0.5,
        }),
        world,
    )
    assert "transient" in coord.intel_pins

    # Before expiry
    before = _minimal_world(tick=3)
    coord._expire_intel_pins(before)
    assert "transient" in coord.intel_pins

    # After expiry
    after = _minimal_world(tick=10)
    coord._expire_intel_pins(after)
    assert "transient" not in coord.intel_pins


# ---------------------------------------------------------------- serialization

def test_serialize_intel_pins_round_trip() -> None:
    coord = _coord()
    world = _minimal_world(tick=42)
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id="s1", data={
            "action": "create", "pin_id": "s1",
            "position": [250.0, 375.0],
            "radius": 300.0, "value": 2.0, "label": "critical",
        }),
        world,
    )
    serialized = coord.serialize_intel_pins()
    assert len(serialized) == 1
    s = serialized[0]
    assert s["pin_id"] == "s1"
    assert s["position"] == [250.0, 375.0]
    assert s["label"] == "critical"
    assert s["value"] == 2.0


# ---------------------------------------------------------------- status

def test_status_summary_reports_pins_and_zones() -> None:
    coord = _coord()
    world = _minimal_world()
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id="a", data={
            "action": "create", "pin_id": "a",
            "position": [100.0, 200.0], "label": "alpha",
        }),
        world,
    )
    summary = coord.build_status_summary(world)
    assert summary["fleet"]["active"] == 1
    assert summary["intel_pins_active"] == [
        {"pin_id": "a", "position": [100.0, 200.0], "label": "alpha"}
    ]
    # Reasonable quadrant bucket present
    assert "quadrants_from_base" in summary


# ---------------------------------------------------------------- market wiring

def test_market_emits_intel_pin_asset() -> None:
    """When a pin is placed, _run_priority_market should include an asset
    with source='intel_pin' keyed by that pin."""
    from src.agents.coordinator import DroneAgent, DroneTask

    # Build a world with a search map so _run_priority_market runs
    base = Vec3(1000.0, 0.0, 1000.0)
    size = 2048
    heightmap = np.zeros((size, size), dtype=np.float32)
    biome_map = np.zeros((size, size), dtype=np.int8)
    terrain = Terrain(
        width=size, height=size, max_elevation=100.0,
        heightmap=heightmap, biome_map=biome_map, survivors=(), seed=0,
    )
    drones = (
        Drone(id=0, position=base, battery=100.0, status=DroneStatus.ACTIVE),
        Drone(id=1, position=Vec3(1200.0, 0.0, 1000.0), battery=100.0, status=DroneStatus.ACTIVE),
    )
    search_map = SearchMap.empty(world_size=float(size), cell_size=40.0)
    search_map.add_gaussian((500.0, 500.0), radius_meters=400.0, weight=1.0)
    search_map.poc += np.float32(0.001)
    search_map.normalize(target_mass=1.0)
    fog = np.full((64, 64), FOG_UNEXPLORED, dtype=np.int8)
    world = WorldState(
        tick=1, elapsed=0.1, terrain=terrain,
        drones=drones, survivors=(), fog_grid=fog, comms_links=(), events=(),
        base_position=base, tick_rate=10.0, search_map=search_map,
    )

    coord = _coord()
    # Need agents populated since _run_priority_market reads them
    for d in drones:
        coord.agents[d.id] = DroneAgent(drone_id=d.id, task=DroneTask.IDLE)
    coord.apply_intel_pin_command(
        Command(type="set_intel_pin", zone_id="pin_market", data={
            "action": "create", "pin_id": "pin_market",
            "position": [1100.0, 1100.0], "radius": 200.0, "value": 1.0,
            "label": "check this",
        }),
        world,
    )

    coord._run_priority_market(world)
    # Some drone should have received the intel pin as its assignment
    # (value 1.0 × source_scale 3.0 = effective 3.0 → beats nearby PoC cells
    # with their much smaller base values).
    pin_assignments = [
        a for a in coord._market_assignments.values()
        if a.source == "intel_pin" and a.asset_id == "intel_pin_market"
    ]
    assert len(pin_assignments) >= 1
