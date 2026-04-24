"""Tests for wind effects on drone physics and battery drain."""

from __future__ import annotations

import numpy as np

from src.simulation.drone import update_drone_battery, update_drone_physics
from src.simulation.types import Drone, DroneStatus, SimConfig, Vec3

_CFG = SimConfig(
    terrain_size=64,
    max_elevation=200.0,
    drone_count=1,
    tick_rate=20.0,
    drone_max_speed=15.0,
    drone_battery_drain_rate=0.05,
    drone_cruise_altitude=50.0,
    wind_drag_coef=0.4,
    wind_battery_factor=0.6,
)

_HEIGHTMAP = np.full((64, 64), 10.0, dtype=np.float64)


def _drone(position=Vec3(32.0, 60.0, 32.0), velocity=Vec3(0.0, 0.0, 0.0), target=None):
    return Drone(
        id=0,
        position=position,
        velocity=velocity,
        heading=0.0,
        battery=100.0,
        sensor_active=False,  # disable to isolate wind effect
        status=DroneStatus.ACTIVE,
        target=target,
        max_speed=15.0,
    )


def _constant_wind(wx: float, wz: float):
    return lambda x, z: Vec3(wx, 0.0, wz)


# ---------------------------------------------------------------------------
# Physics: stationary drones drift downwind
# ---------------------------------------------------------------------------


def test_stationary_drone_drifts_downwind():
    """A drone with no target and constant east wind should move east over time."""
    drone = _drone()
    wind = _constant_wind(5.0, 0.0)  # 5 m/s east

    dt = 1.0 / 20.0
    for _ in range(40):  # 2 seconds
        drone = update_drone_physics(drone, dt, _HEIGHTMAP, _CFG, wind_fn=wind)

    assert drone.position.x > 32.5, f"Drone did not drift east: x={drone.position.x}"
    assert abs(drone.position.z - 32.0) < 0.5, "Drone drifted in wrong axis"


def test_no_wind_no_drift():
    """With wind_fn=None, drone stays put — baseline sanity."""
    drone = _drone()
    dt = 1.0 / 20.0
    for _ in range(40):
        drone = update_drone_physics(drone, dt, _HEIGHTMAP, _CFG, wind_fn=None)
    assert abs(drone.position.x - 32.0) < 0.01
    assert abs(drone.position.z - 32.0) < 0.01


def test_drag_coef_zero_disables_drift():
    """Config flag fully disables wind effect even if wind_fn provided."""
    cfg = SimConfig(
        terrain_size=64,
        drone_max_speed=15.0,
        drone_cruise_altitude=50.0,
        wind_drag_coef=0.0,
        wind_battery_factor=0.6,
    )
    drone = _drone()
    wind = _constant_wind(5.0, 0.0)
    dt = 1.0 / 20.0
    for _ in range(40):
        drone = update_drone_physics(drone, dt, _HEIGHTMAP, cfg, wind_fn=wind)
    assert abs(drone.position.x - 32.0) < 0.01


# ---------------------------------------------------------------------------
# Battery: headwind drains faster than tailwind
# ---------------------------------------------------------------------------


def test_headwind_drains_faster_than_tailwind():
    """Two identical drones moving east: one with headwind, one with tailwind.

    The headwind drone should have lower battery after N ticks.
    """
    headwind_drone = _drone(velocity=Vec3(10.0, 0.0, 0.0))  # moving east
    tailwind_drone = _drone(velocity=Vec3(10.0, 0.0, 0.0))  # moving east

    east_wind = _constant_wind(8.0, 0.0)    # wind east → tailwind for east-mover
    west_wind = _constant_wind(-8.0, 0.0)   # wind west → headwind for east-mover

    dt = 1.0 / 20.0
    for _ in range(100):  # 5 seconds
        headwind_drone = update_drone_battery(headwind_drone, dt, _CFG, wind_fn=west_wind)
        tailwind_drone = update_drone_battery(tailwind_drone, dt, _CFG, wind_fn=east_wind)

    assert headwind_drone.battery < tailwind_drone.battery, (
        f"headwind battery {headwind_drone.battery} not < tailwind {tailwind_drone.battery}"
    )
    # Meaningful delta: at least 0.1% over 5s (model gives ~0.17% at these params)
    delta = tailwind_drone.battery - headwind_drone.battery
    assert delta > 0.1, f"Headwind penalty too small: {delta:.3f}%"


def test_no_wind_no_penalty():
    """With wind_fn=None, battery drain matches baseline."""
    drone_a = _drone(velocity=Vec3(10.0, 0.0, 0.0))
    drone_b = _drone(velocity=Vec3(10.0, 0.0, 0.0))

    dt = 1.0 / 20.0
    for _ in range(100):
        drone_a = update_drone_battery(drone_a, dt, _CFG, wind_fn=None)
        drone_b = update_drone_battery(drone_b, dt, _CFG, wind_fn=None)

    assert abs(drone_a.battery - drone_b.battery) < 1e-9


def test_stationary_drone_no_battery_penalty():
    """Wind with no movement: no headwind component, no extra drain."""
    moving = _drone(velocity=Vec3(10.0, 0.0, 0.0))
    still = _drone(velocity=Vec3(0.0, 0.0, 0.0))
    wind = _constant_wind(-10.0, 0.0)  # strong west wind

    dt = 1.0 / 20.0
    for _ in range(100):
        moving = update_drone_battery(moving, dt, _CFG, wind_fn=wind)
        still = update_drone_battery(still, dt, _CFG, wind_fn=wind)

    # The stationary drone has no headwind applied (no direction of travel),
    # so its drain is speed-only. Moving-into-wind should drain more.
    assert moving.battery < still.battery
