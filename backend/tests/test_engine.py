"""Tests for the core simulation engine."""

from __future__ import annotations

import numpy as np

from src.simulation.engine import create_world, get_coverage_pct, tick
from src.simulation.types import (
    FOG_EXPLORED,
    FOG_UNEXPLORED,
    Command,
    DroneStatus,
    EventType,
    SimConfig,
    Vec3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = SimConfig(
    terrain_size=64,
    terrain_seed=42,
    max_elevation=200.0,
    drone_count=4,
    tick_rate=20.0,
    survivor_count=5,
    drone_max_speed=15.0,
    drone_sensor_range=40.0,
    drone_comms_range=100.0,
    drone_battery_drain_rate=0.05,
    drone_battery_critical=15.0,
    drone_cruise_altitude=50.0,
    # Disable random failures for deterministic tests
    sensor_failure_prob=0.0,
    comms_failure_prob=0.0,
)


def _tick_n(world, n, config=None):
    """Run n ticks with default dt, no commands, no rng."""
    cfg = config or _SMALL_CFG
    dt = 1.0 / cfg.tick_rate
    for _ in range(n):
        world = tick(world, dt, config=cfg)
    return world


# ---------------------------------------------------------------------------
# create_world
# ---------------------------------------------------------------------------


def test_create_world_tick_zero():
    world = create_world(_SMALL_CFG)
    assert world.tick == 0
    assert world.elapsed == 0.0


def test_create_world_terrain_present():
    world = create_world(_SMALL_CFG)
    assert world.terrain is not None
    assert world.terrain.width == _SMALL_CFG.terrain_size
    assert world.terrain.height == _SMALL_CFG.terrain_size


def test_create_world_correct_drone_count():
    world = create_world(_SMALL_CFG)
    assert len(world.drones) == _SMALL_CFG.drone_count


def test_create_world_correct_survivor_count():
    world = create_world(_SMALL_CFG)
    assert len(world.survivors) == _SMALL_CFG.survivor_count


def test_create_world_fog_all_unexplored():
    world = create_world(_SMALL_CFG)
    fog = world.fog_grid
    assert fog.shape == (_SMALL_CFG.terrain_size, _SMALL_CFG.terrain_size)
    assert np.all(fog == FOG_UNEXPLORED)


def test_create_world_base_position():
    world = create_world(_SMALL_CFG)
    expected_x = _SMALL_CFG.terrain_size / 2.0
    expected_z = _SMALL_CFG.terrain_size / 2.0
    assert world.base_position.x == expected_x
    assert world.base_position.z == expected_z


def test_create_world_no_events():
    world = create_world(_SMALL_CFG)
    assert len(world.events) == 0


def test_create_world_drones_all_active():
    world = create_world(_SMALL_CFG)
    for d in world.drones:
        assert d.status == DroneStatus.ACTIVE
        assert d.battery == 100.0


# ---------------------------------------------------------------------------
# tick — basic progression
# ---------------------------------------------------------------------------


def test_tick_advances_tick_counter():
    world = create_world(_SMALL_CFG)
    dt = 1.0 / _SMALL_CFG.tick_rate
    world2 = tick(world, dt, config=_SMALL_CFG)
    assert world2.tick == 1
    assert world2.elapsed > 0.0


def test_tick_elapsed_accumulates():
    world = create_world(_SMALL_CFG)
    dt = 1.0 / _SMALL_CFG.tick_rate
    world = tick(world, dt, config=_SMALL_CFG)
    world = tick(world, dt, config=_SMALL_CFG)
    assert world.tick == 2
    assert abs(world.elapsed - 2 * dt) < 1e-9


def test_tick_drones_gain_altitude():
    """Drones start at y=0 and should rise toward cruise altitude over time."""
    world = create_world(_SMALL_CFG)
    initial_y = max(d.position.y for d in world.drones)

    world = _tick_n(world, 50)

    final_y = max(d.position.y for d in world.drones)
    assert final_y > initial_y, "Drones should rise toward cruise altitude"


def test_tick_fog_updates():
    """After some ticks, the fog grid should have some explored cells."""
    world = create_world(_SMALL_CFG)
    assert np.all(world.fog_grid == FOG_UNEXPLORED)

    world = _tick_n(world, 5)

    explored_count = np.count_nonzero(world.fog_grid == FOG_EXPLORED)
    assert explored_count > 0, "Some fog cells should be explored after ticks"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_move_to_command_changes_target():
    world = create_world(_SMALL_CFG)
    target = Vec3(10.0, 0.0, 10.0)
    cmd = Command(type="move_to", drone_id=0, target=target)

    dt = 1.0 / _SMALL_CFG.tick_rate
    world2 = tick(world, dt, commands=[cmd], config=_SMALL_CFG)

    # After applying the command and one physics step, the drone should have a target
    # and should be moving (or at least have a non-idle task)
    drone = world2.drones[0]
    assert drone.target is not None
    assert drone.current_task == "moving"


def test_move_to_drone_moves_toward_target():
    world = create_world(_SMALL_CFG)
    target = Vec3(10.0, 0.0, 10.0)
    cmd = Command(type="move_to", drone_id=0, target=target)

    dt = 1.0 / _SMALL_CFG.tick_rate
    initial_pos = world.drones[0].position

    # Apply command and run several ticks
    world = tick(world, dt, commands=[cmd], config=_SMALL_CFG)
    world = _tick_n(world, 40)

    final_pos = world.drones[0].position
    initial_dist = ((initial_pos.x - target.x) ** 2 + (initial_pos.z - target.z) ** 2) ** 0.5
    final_dist = ((final_pos.x - target.x) ** 2 + (final_pos.z - target.z) ** 2) ** 0.5
    assert final_dist < initial_dist, "Drone should move closer to target"


def test_command_ignored_for_failed_drone():
    world = create_world(_SMALL_CFG)
    # Manually fail a drone
    drones = list(world.drones)
    from dataclasses import replace

    drones[0] = replace(drones[0], status=DroneStatus.FAILED)
    world = replace(world, drones=tuple(drones))

    target = Vec3(10.0, 0.0, 10.0)
    cmd = Command(type="move_to", drone_id=0, target=target)
    dt = 1.0 / _SMALL_CFG.tick_rate
    world2 = tick(world, dt, commands=[cmd], config=_SMALL_CFG)

    assert world2.drones[0].target is None


def test_return_to_base_command():
    world = create_world(_SMALL_CFG)
    cmd = Command(type="return_to_base", drone_id=0)
    dt = 1.0 / _SMALL_CFG.tick_rate
    world2 = tick(world, dt, commands=[cmd], config=_SMALL_CFG)

    drone = world2.drones[0]
    assert drone.status == DroneStatus.RETURNING
    assert drone.current_task == "returning"


# ---------------------------------------------------------------------------
# Survivor discovery
# ---------------------------------------------------------------------------


def test_survivor_discovery_event():
    """A drone placed directly on a survivor should trigger SURVIVOR_FOUND."""
    world = create_world(_SMALL_CFG)

    if len(world.survivors) == 0:
        return  # skip if no survivors

    # Move a drone right on top of the first survivor
    survivor = world.survivors[0]
    from dataclasses import replace

    drones = list(world.drones)
    drones[0] = replace(
        drones[0],
        position=Vec3(survivor.position.x, survivor.position.y + 30.0, survivor.position.z),
    )
    world = replace(world, drones=tuple(drones))

    dt = 1.0 / _SMALL_CFG.tick_rate
    world2 = tick(world, dt, config=_SMALL_CFG)

    # Check that the survivor was discovered
    found_events = [e for e in world2.events if e.type == EventType.SURVIVOR_FOUND]
    assert len(found_events) >= 1
    assert found_events[0].survivor_id == 0

    # Survivor should be marked discovered in the new world state
    assert world2.survivors[0].discovered is True
    assert world2.survivors[0].discovered_by == drones[0].id


def test_survivor_not_rediscovered():
    """A survivor that's already discovered should not trigger another event."""
    world = create_world(_SMALL_CFG)

    if len(world.survivors) == 0:
        return

    survivor = world.survivors[0]
    from dataclasses import replace

    # Pre-mark as discovered
    survivors = list(world.survivors)
    survivors[0] = replace(survivors[0], discovered=True, discovered_by=1)
    drones = list(world.drones)
    drones[0] = replace(
        drones[0],
        position=Vec3(survivor.position.x, survivor.position.y + 30.0, survivor.position.z),
    )
    world = replace(world, drones=tuple(drones), survivors=tuple(survivors))

    dt = 1.0 / _SMALL_CFG.tick_rate
    world2 = tick(world, dt, config=_SMALL_CFG)

    # Survivor 0 specifically should not be re-discovered (other survivors may be found)
    found_events_s0 = [
        e for e in world2.events if e.type == EventType.SURVIVOR_FOUND and e.survivor_id == 0
    ]
    assert len(found_events_s0) == 0


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_starts_at_zero():
    world = create_world(_SMALL_CFG)
    assert get_coverage_pct(world.fog_grid) == 0.0


def test_coverage_increases_over_time():
    world = create_world(_SMALL_CFG)
    cov0 = get_coverage_pct(world.fog_grid)

    world = _tick_n(world, 10)

    cov1 = get_coverage_pct(world.fog_grid)
    assert cov1 > cov0, "Coverage should increase as drones explore"


def test_coverage_pct_calculation():
    """Verify get_coverage_pct with a manually crafted fog grid."""
    grid = np.full((10, 10), FOG_UNEXPLORED, dtype=np.int8)
    assert get_coverage_pct(grid) == 0.0

    # Mark 25 of 100 cells as explored
    grid[:5, :5] = FOG_EXPLORED
    assert abs(get_coverage_pct(grid) - 25.0) < 1e-9

    # Mark all cells
    grid[:] = FOG_EXPLORED
    assert abs(get_coverage_pct(grid) - 100.0) < 1e-9


def test_coverage_never_exceeds_100():
    world = create_world(_SMALL_CFG)
    world = _tick_n(world, 200)
    assert get_coverage_pct(world.fog_grid) <= 100.0


# ---------------------------------------------------------------------------
# Communication links in engine context
# ---------------------------------------------------------------------------


def test_comms_links_populated_after_tick():
    """Drones starting near each other at the base should have comms links."""
    world = create_world(_SMALL_CFG)
    dt = 1.0 / _SMALL_CFG.tick_rate
    world = tick(world, dt, config=_SMALL_CFG)

    # All 4 drones start near the base within comms range of each other
    assert len(world.comms_links) > 0


# ---------------------------------------------------------------------------
# Battery depletion in engine context
# ---------------------------------------------------------------------------


def test_battery_critical_event():
    """A drone with low battery should generate DRONE_BATTERY_CRITICAL event."""
    # Use a high drain rate so battery drops below critical in one tick
    high_drain_cfg = SimConfig(
        terrain_size=64,
        terrain_seed=42,
        max_elevation=200.0,
        drone_count=4,
        tick_rate=20.0,
        survivor_count=5,
        drone_battery_drain_rate=5.0,  # very high drain
        drone_battery_critical=15.0,
        drone_cruise_altitude=50.0,
        sensor_failure_prob=0.0,
        comms_failure_prob=0.0,
    )
    world = create_world(high_drain_cfg)
    from dataclasses import replace

    drones = list(world.drones)
    # Set battery just above critical; high drain rate ensures it drops below in one tick.
    # Drain per tick ~ 5.0 * 1.0 * 1.0 * 0.05 * 1.2 = 0.3, so start just 0.1 above.
    # Also move drone far from base so it doesn't immediately switch to RECHARGING.
    drones[0] = replace(
        drones[0],
        battery=high_drain_cfg.drone_battery_critical + 0.1,
        position=Vec3(10.0, 0.0, 10.0),
    )
    world = replace(world, drones=tuple(drones))

    dt = 1.0 / high_drain_cfg.tick_rate
    world2 = tick(world, dt, config=high_drain_cfg)

    critical_events = [e for e in world2.events if e.type == EventType.DRONE_BATTERY_CRITICAL]
    assert len(critical_events) >= 1
    assert critical_events[0].drone_id == 0

    # Drone should be RETURNING and targeting base
    assert world2.drones[0].status == DroneStatus.RETURNING
    assert world2.drones[0].target == world.base_position
