"""Tests for the SwarmCoordinator agent layer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.agents.coordinator import (
    DroneTask,
    PriorityZone,
    SearchPhase,
    SwarmCoordinator,
)
from src.simulation.engine import create_world, tick
from src.simulation.types import (
    FOG_EXPLORED,
    FOG_UNEXPLORED,
    Command,
    DroneStatus,
    SimConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = SimConfig(
    terrain_size=64,
    terrain_seed=42,
    drone_count=5,
    survivor_count=5,
    sensor_failure_prob=0.0,
    comms_failure_prob=0.0,
)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_coordinator_initializes_with_correct_phase():
    coord = SwarmCoordinator(_CFG)
    assert coord.phase == SearchPhase.INITIAL_SPREAD
    assert coord.tick_count == 0
    assert coord.zones_assigned is False


# ---------------------------------------------------------------------------
# update — basic behaviour
# ---------------------------------------------------------------------------


def test_coordinator_skips_holding_drones():
    """A drone with current_task='holding' must not be reassigned by the
    coordinator — the human override is authoritative until a new command."""
    world = create_world(_CFG)
    drones = list(world.drones)
    drones[0] = replace(drones[0], current_task="holding")
    world = replace(world, drones=tuple(drones))

    coord = SwarmCoordinator(_CFG)
    commands = coord.update(world, _CFG)

    commanded_ids = {cmd.drone_id for cmd in commands}
    assert 0 not in commanded_ids, "coordinator should skip drone 0 while it is holding"


def test_update_returns_commands_for_active_drones():
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)
    commands = coord.update(world, _CFG)

    # All 5 drones are active, so we should get commands for them
    commanded_ids = {cmd.drone_id for cmd in commands}
    active_ids = {d.id for d in world.drones if d.status == DroneStatus.ACTIVE}
    assert commanded_ids == active_ids


def test_update_assigns_zones_on_first_call():
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)
    coord.update(world, _CFG)

    assert coord.zones_assigned is True
    # Every active drone should have a zone and MOVING_TO_ZONE task
    for drone in world.drones:
        if drone.status == DroneStatus.ACTIVE:
            agent = coord.agents[drone.id]
            assert agent.zone is not None
            assert agent.task == DroneTask.MOVING_TO_ZONE


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


def test_phase_transitions_on_coverage_milestones():
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)

    # Run once to bootstrap zones
    coord.update(world, _CFG)

    # Manually set fog to simulate coverage levels and re-run update
    fog = np.array(world.fog_grid, copy=True)
    total = fog.size

    # 6% explored -> should move to SYSTEMATIC_SEARCH
    explore_count = int(total * 0.06)
    fog.flat[:explore_count] = FOG_EXPLORED
    world_6 = replace(world, fog_grid=fog.copy())
    coord.update(world_6, _CFG)
    assert coord.phase == SearchPhase.SYSTEMATIC_SEARCH

    # 72% explored -> should move to FRONTIER_HUNT
    explore_count = int(total * 0.72)
    fog.flat[:explore_count] = FOG_EXPLORED
    world_72 = replace(world, fog_grid=fog.copy())
    coord.update(world_72, _CFG)
    assert coord.phase == SearchPhase.FRONTIER_HUNT

    # 91% explored -> PRIORITY_SWEEP
    explore_count = int(total * 0.91)
    fog.flat[:explore_count] = FOG_EXPLORED
    world_91 = replace(world, fog_grid=fog.copy())
    coord.update(world_91, _CFG)
    assert coord.phase == SearchPhase.PRIORITY_SWEEP

    # 100% explored -> COMPLETE
    fog[:] = FOG_EXPLORED
    world_100 = replace(world, fog_grid=fog.copy())
    coord.update(world_100, _CFG)
    assert coord.phase == SearchPhase.COMPLETE


# ---------------------------------------------------------------------------
# Failed drones
# ---------------------------------------------------------------------------


def test_failed_drones_do_not_receive_commands():
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)

    # Fail drone 0
    drones = list(world.drones)
    drones[0] = replace(drones[0], status=DroneStatus.FAILED)
    world = replace(world, drones=tuple(drones))

    commands = coord.update(world, _CFG)
    commanded_ids = {cmd.drone_id for cmd in commands}
    assert 0 not in commanded_ids


# ---------------------------------------------------------------------------
# Integration: coordinator + engine loop
# ---------------------------------------------------------------------------


def test_coordinator_with_engine_runs_100_ticks():
    """Run the coordinator feeding commands into the sim engine for 100 ticks
    and verify no exceptions are raised and that basic invariants hold."""
    config = SimConfig(
        terrain_size=64,
        terrain_seed=42,
        drone_count=5,
        survivor_count=5,
        sensor_failure_prob=0.0,
        comms_failure_prob=0.0,
    )
    world = create_world(config)
    coord = SwarmCoordinator(config)
    dt = 1.0 / config.tick_rate

    for _ in range(100):
        commands = coord.update(world, config)
        world = tick(world, dt, commands=commands, config=config)

    assert world.tick == 100
    # Coverage should have increased from zero
    fog = np.asarray(world.fog_grid)
    explored = np.count_nonzero(fog != FOG_UNEXPLORED)
    assert explored > 0, "Some terrain should be explored after 100 ticks"

    # All drones should still be in a valid status
    for d in world.drones:
        assert d.status in (
            DroneStatus.ACTIVE,
            DroneStatus.RETURNING,
            DroneStatus.RECHARGING,
            DroneStatus.FAILED,
        )


# ---------------------------------------------------------------------------
# Priority zones
# ---------------------------------------------------------------------------


def _zone_command(
    action: str,
    zone_id: str | None = None,
    polygon: list[list[float]] | None = None,
    priority: str | None = None,
) -> Command:
    data: dict = {"action": action}
    if zone_id is not None:
        data["zone_id"] = zone_id
    if polygon is not None:
        data["polygon"] = polygon
    if priority is not None:
        data["priority"] = priority
    return Command(
        type="set_priority",
        zone_id=zone_id,
        priority=priority,
        data=data,
    )


def test_zone_create_update_delete():
    coord = SwarmCoordinator(_CFG)
    world = create_world(_CFG)

    poly = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]
    coord.apply_zone_command(
        _zone_command("create", "z1", poly, "high"),
        world,
    )
    assert "z1" in coord.zones
    assert coord.zones["z1"].priority == "high"

    # Update priority without polygon
    coord.apply_zone_command(
        _zone_command("update", "z1", priority="low"),
        world,
    )
    assert coord.zones["z1"].priority == "low"

    # Delete
    coord.apply_zone_command(_zone_command("delete", "z1"), world)
    assert "z1" not in coord.zones


def test_zone_multiplier_bias():
    coord = SwarmCoordinator(_CFG)
    coord.zones["high"] = PriorityZone(
        "high",
        ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        "high",
        0,
    )
    coord.zones["avoid"] = PriorityZone(
        "avoid",
        ((500.0, 500.0), (600.0, 500.0), (600.0, 600.0), (500.0, 600.0)),
        "avoid",
        0,
    )
    coord.zones["low"] = PriorityZone(
        "low",
        ((1000.0, 0.0), (1100.0, 0.0), (1100.0, 100.0), (1000.0, 100.0)),
        "low",
        0,
    )

    # Inside high zone → 3.0
    assert coord._zone_multiplier(50.0, 50.0) == 3.0
    # Inside avoid zone → 0.0 (drones must skip)
    assert coord._zone_multiplier(550.0, 550.0) == 0.0
    # Inside low zone → 0.3
    assert coord._zone_multiplier(1050.0, 50.0) == 0.3
    # Outside any zone → 1.0
    assert coord._zone_multiplier(2000.0, 2000.0) == 1.0


def test_zone_clear_wipes_all():
    coord = SwarmCoordinator(_CFG)
    world = create_world(_CFG)
    poly = [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]
    coord.apply_zone_command(_zone_command("create", "a", poly, "high"), world)
    coord.apply_zone_command(_zone_command("create", "b", poly, "avoid"), world)
    assert len(coord.zones) == 2
    coord.apply_zone_command(_zone_command("clear"), world)
    assert coord.zones == {}


def test_zone_invalid_priority_ignored():
    coord = SwarmCoordinator(_CFG)
    world = create_world(_CFG)
    poly = [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]
    coord.apply_zone_command(
        _zone_command("create", "bad", poly, "bogus"),
        world,
    )
    assert "bad" not in coord.zones


def test_serialize_zones_round_trip():
    coord = SwarmCoordinator(_CFG)
    world = create_world(_CFG)
    poly = [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]
    coord.apply_zone_command(_zone_command("create", "z", poly, "high"), world)
    payload = coord.serialize_zones()
    assert len(payload) == 1
    assert payload[0]["zone_id"] == "z"
    assert payload[0]["priority"] == "high"
    assert payload[0]["polygon"] == poly
