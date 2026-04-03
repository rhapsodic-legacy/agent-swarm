"""Tests for drone physics, detection, and communication."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.simulation.drone import (
    compute_comms_links,
    create_drone_fleet,
    detect_survivors,
    update_drone_battery,
    update_drone_physics,
)
from src.simulation.types import (
    Drone,
    DroneStatus,
    SimConfig,
    Survivor,
    Vec3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = SimConfig(
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
)

_BASE = Vec3(32.0, 0.0, 32.0)

# A flat heightmap at elevation 10 for easy physics tests.
_FLAT_HEIGHTMAP = np.full((64, 64), 10.0, dtype=np.float64)


def _make_drone(**overrides) -> Drone:
    """Create a single drone with sensible defaults, overriding as needed."""
    defaults = dict(
        id=0,
        position=Vec3(32.0, 60.0, 32.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        heading=0.0,
        battery=100.0,
        sensor_active=True,
        comms_active=True,
        status=DroneStatus.ACTIVE,
        current_task="idle",
        target=None,
        max_speed=15.0,
        sensor_range=40.0,
        comms_range=100.0,
    )
    defaults.update(overrides)
    return Drone(**defaults)


# ---------------------------------------------------------------------------
# create_drone_fleet
# ---------------------------------------------------------------------------


def test_fleet_correct_count():
    fleet = create_drone_fleet(4, _BASE, _CFG)
    assert len(fleet) == 4


def test_fleet_positions_near_base():
    fleet = create_drone_fleet(4, _BASE, _CFG)
    for d in fleet:
        dist = (d.position - _BASE).length()
        assert dist < 20.0, f"Drone {d.id} too far from base: {dist:.1f}m"


def test_fleet_all_full_battery():
    fleet = create_drone_fleet(4, _BASE, _CFG)
    for d in fleet:
        assert d.battery == 100.0


def test_fleet_all_active():
    fleet = create_drone_fleet(4, _BASE, _CFG)
    for d in fleet:
        assert d.status == DroneStatus.ACTIVE


def test_fleet_unique_ids():
    fleet = create_drone_fleet(10, _BASE, _CFG)
    ids = [d.id for d in fleet]
    assert len(ids) == len(set(ids))


def test_fleet_uses_config_speeds():
    cfg = SimConfig(drone_max_speed=25.0, drone_sensor_range=60.0, drone_comms_range=200.0)
    fleet = create_drone_fleet(2, _BASE, cfg)
    for d in fleet:
        assert d.max_speed == 25.0
        assert d.sensor_range == 60.0
        assert d.comms_range == 200.0


# ---------------------------------------------------------------------------
# update_drone_physics
# ---------------------------------------------------------------------------


def test_drone_moves_toward_target():
    """A drone with a target should move closer to it after a physics step."""
    target = Vec3(50.0, 60.0, 32.0)
    drone = _make_drone(target=target)
    initial_dist = (drone.position - target).length_xz()

    # Run several steps so acceleration has time to take effect
    for _ in range(10):
        drone = update_drone_physics(drone, 0.05, _FLAT_HEIGHTMAP, _CFG)

    new_dist = (drone.position - target).length_xz()
    assert new_dist < initial_dist, "Drone should move closer to target"


def test_drone_stays_within_terrain_bounds():
    """A drone moving toward a position outside the map should be clamped."""
    target = Vec3(200.0, 60.0, 200.0)  # well outside 64x64
    drone = _make_drone(target=target, position=Vec3(62.0, 60.0, 62.0))

    for _ in range(100):
        drone = update_drone_physics(drone, 0.05, _FLAT_HEIGHTMAP, _CFG)

    assert 0.0 <= drone.position.x <= 63.0
    assert 0.0 <= drone.position.z <= 63.0


def test_drone_maintains_cruise_altitude():
    """After many steps, drone altitude should converge near cruise altitude above terrain."""
    drone = _make_drone(position=Vec3(32.0, 0.0, 32.0))
    for _ in range(200):
        drone = update_drone_physics(drone, 0.05, _FLAT_HEIGHTMAP, _CFG)

    expected_alt = 10.0 + _CFG.drone_cruise_altitude  # terrain + cruise
    assert abs(drone.position.y - expected_alt) < 2.0, (
        f"Expected altitude ~{expected_alt}, got {drone.position.y}"
    )


def test_drone_no_target_stays_still():
    """A drone with no target and zero velocity should stay in place (horizontally)."""
    drone = _make_drone(target=None, position=Vec3(32.0, 60.0, 32.0))
    original_x, original_z = drone.position.x, drone.position.z

    for _ in range(20):
        drone = update_drone_physics(drone, 0.05, _FLAT_HEIGHTMAP, _CFG)

    assert abs(drone.position.x - original_x) < 0.1
    assert abs(drone.position.z - original_z) < 0.1


def test_failed_drone_does_not_move():
    """A FAILED drone should remain stationary and have zero velocity."""
    drone = _make_drone(
        status=DroneStatus.FAILED,
        target=Vec3(50.0, 60.0, 50.0),
        velocity=Vec3(5.0, 0.0, 5.0),
    )
    original_pos = drone.position

    drone = update_drone_physics(drone, 0.05, _FLAT_HEIGHTMAP, _CFG)

    assert drone.position == original_pos
    assert drone.velocity == Vec3(0.0, 0.0, 0.0)


def test_drone_altitude_above_terrain():
    """Drone should never go below terrain surface."""
    # Create a heightmap with a bump
    hmap = np.full((64, 64), 10.0, dtype=np.float64)
    hmap[30:35, 30:35] = 150.0  # tall hill

    drone = _make_drone(position=Vec3(32.0, 20.0, 32.0))  # start low near the hill
    drone = update_drone_physics(drone, 0.05, hmap, _CFG)

    ix = int(min(max(drone.position.x, 0.0), 63))
    iz = int(min(max(drone.position.z, 0.0), 63))
    terrain_elev = float(hmap[iz, ix])
    assert drone.position.y >= terrain_elev - 0.01


# ---------------------------------------------------------------------------
# update_drone_battery
# ---------------------------------------------------------------------------


def test_battery_drains_over_time():
    drone = _make_drone(battery=100.0)
    drone = update_drone_battery(drone, 1.0, _CFG)
    assert drone.battery < 100.0


def test_battery_returns_when_critical():
    drone = _make_drone(battery=_CFG.drone_battery_critical + 0.01)
    # Drain enough to go below critical
    drone = update_drone_battery(drone, 10.0, _CFG)
    assert drone.status == DroneStatus.RETURNING


def test_battery_fails_when_depleted():
    drone = _make_drone(battery=0.01)
    drone = update_drone_battery(drone, 100.0, _CFG)
    assert drone.battery == 0.0
    assert drone.status == DroneStatus.FAILED


def test_battery_does_not_go_negative():
    drone = _make_drone(battery=0.001)
    drone = update_drone_battery(drone, 1000.0, _CFG)
    assert drone.battery >= 0.0


def test_failed_drone_battery_unchanged():
    drone = _make_drone(battery=50.0, status=DroneStatus.FAILED)
    result = update_drone_battery(drone, 1.0, _CFG)
    assert result.battery == 50.0
    assert result is drone  # same object, no change


def test_recharging_drone_battery_unchanged():
    drone = _make_drone(battery=50.0, status=DroneStatus.RECHARGING)
    result = update_drone_battery(drone, 1.0, _CFG)
    assert result.battery == 50.0
    assert result is drone


def test_battery_drain_faster_with_sensor():
    d_sensor_on = _make_drone(battery=100.0, sensor_active=True)
    d_sensor_off = _make_drone(battery=100.0, sensor_active=False)

    d_sensor_on = update_drone_battery(d_sensor_on, 1.0, _CFG)
    d_sensor_off = update_drone_battery(d_sensor_off, 1.0, _CFG)

    assert d_sensor_on.battery < d_sensor_off.battery


# ---------------------------------------------------------------------------
# detect_survivors
# ---------------------------------------------------------------------------


def test_detect_nearby_survivor():
    drone = _make_drone(position=Vec3(30.0, 60.0, 30.0), sensor_range=40.0)
    survivors = (
        Survivor(id=0, position=Vec3(31.0, 10.0, 31.0)),  # ~1.4m away (XZ)
    )
    detected = detect_survivors(drone, survivors)
    assert 0 in detected


def test_no_detect_far_survivor():
    drone = _make_drone(position=Vec3(30.0, 60.0, 30.0), sensor_range=40.0)
    survivors = (
        Survivor(id=0, position=Vec3(0.0, 10.0, 0.0)),  # ~42m away (XZ)
    )
    detected = detect_survivors(drone, survivors)
    assert 0 not in detected


def test_detect_uses_80_percent_threshold():
    """Detection is guaranteed within 80% of sensor range, impossible beyond."""
    sensor_range = 40.0
    drone = _make_drone(position=Vec3(0.0, 60.0, 0.0), sensor_range=sensor_range)

    # Just inside 80% (31m < 32m threshold)
    s_inside = Survivor(id=0, position=Vec3(31.0, 10.0, 0.0))
    # Just outside 80% (33m > 32m threshold)
    s_outside = Survivor(id=1, position=Vec3(33.0, 10.0, 0.0))

    detected = detect_survivors(drone, (s_inside, s_outside))
    assert 0 in detected
    assert 1 not in detected


def test_no_detect_already_discovered():
    drone = _make_drone(position=Vec3(30.0, 60.0, 30.0))
    survivors = (Survivor(id=0, position=Vec3(31.0, 10.0, 31.0), discovered=True),)
    detected = detect_survivors(drone, survivors)
    assert len(detected) == 0


def test_no_detect_when_sensor_inactive():
    drone = _make_drone(position=Vec3(30.0, 60.0, 30.0), sensor_active=False)
    survivors = (Survivor(id=0, position=Vec3(31.0, 10.0, 31.0)),)
    detected = detect_survivors(drone, survivors)
    assert len(detected) == 0


def test_no_detect_when_failed():
    drone = _make_drone(
        position=Vec3(30.0, 60.0, 30.0),
        status=DroneStatus.FAILED,
    )
    survivors = (Survivor(id=0, position=Vec3(31.0, 10.0, 31.0)),)
    detected = detect_survivors(drone, survivors)
    assert len(detected) == 0


def test_detect_multiple_survivors():
    drone = _make_drone(position=Vec3(30.0, 60.0, 30.0), sensor_range=40.0)
    survivors = (
        Survivor(id=0, position=Vec3(31.0, 10.0, 31.0)),
        Survivor(id=1, position=Vec3(29.0, 10.0, 29.0)),
        Survivor(id=2, position=Vec3(0.0, 10.0, 0.0)),  # too far
    )
    detected = detect_survivors(drone, survivors)
    assert set(detected) == {0, 1}


# ---------------------------------------------------------------------------
# compute_comms_links
# ---------------------------------------------------------------------------


def test_comms_links_in_range():
    d0 = _make_drone(id=0, position=Vec3(10.0, 60.0, 10.0), comms_range=100.0)
    d1 = _make_drone(id=1, position=Vec3(20.0, 60.0, 10.0), comms_range=100.0)
    links = compute_comms_links((d0, d1))
    assert (0, 1) in links


def test_comms_links_out_of_range():
    d0 = _make_drone(id=0, position=Vec3(0.0, 60.0, 0.0), comms_range=100.0)
    d1 = _make_drone(id=1, position=Vec3(200.0, 60.0, 0.0), comms_range=100.0)
    links = compute_comms_links((d0, d1))
    assert len(links) == 0


def test_comms_links_failed_drone_excluded():
    d0 = _make_drone(id=0, position=Vec3(10.0, 60.0, 10.0), status=DroneStatus.FAILED)
    d1 = _make_drone(id=1, position=Vec3(11.0, 60.0, 10.0))
    links = compute_comms_links((d0, d1))
    assert len(links) == 0


def test_comms_links_inactive_comms_excluded():
    d0 = _make_drone(id=0, position=Vec3(10.0, 60.0, 10.0), comms_active=False)
    d1 = _make_drone(id=1, position=Vec3(11.0, 60.0, 10.0))
    links = compute_comms_links((d0, d1))
    assert len(links) == 0


def test_comms_links_sorted_ids():
    """Links should always have the lower ID first."""
    d0 = _make_drone(id=5, position=Vec3(10.0, 60.0, 10.0))
    d1 = _make_drone(id=2, position=Vec3(11.0, 60.0, 10.0))
    links = compute_comms_links((d0, d1))
    assert links == ((2, 5),)


def test_comms_links_three_drones():
    d0 = _make_drone(id=0, position=Vec3(0.0, 60.0, 0.0), comms_range=100.0)
    d1 = _make_drone(id=1, position=Vec3(50.0, 60.0, 0.0), comms_range=100.0)
    d2 = _make_drone(id=2, position=Vec3(200.0, 60.0, 0.0), comms_range=100.0)
    links = compute_comms_links((d0, d1, d2))
    # 0-1 are in range (50m), 1-2 are not (150m), 0-2 are not (200m)
    assert links == ((0, 1),)


def test_comms_uses_min_range():
    """Communication uses the minimum of the two drones' comms ranges."""
    d0 = _make_drone(id=0, position=Vec3(0.0, 60.0, 0.0), comms_range=30.0)
    d1 = _make_drone(id=1, position=Vec3(25.0, 60.0, 0.0), comms_range=100.0)
    links = compute_comms_links((d0, d1))
    # Distance 25m, but min range is 30m, so in range
    assert (0, 1) in links

    d0_short = replace(d0, comms_range=20.0)
    links2 = compute_comms_links((d0_short, d1))
    # Distance 25m > min range 20m
    assert len(links2) == 0
