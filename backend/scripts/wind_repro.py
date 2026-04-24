"""End-to-end wind repro: spawn a sim, tick N seconds, confirm wind moves a drone.

Run from the backend directory:
    uv run python scripts/wind_repro.py
"""
from __future__ import annotations

import numpy as np

from src.simulation.drone import update_drone_physics
from src.simulation.types import Drone, DroneStatus, SimConfig, Vec3
from src.simulation.weather import WeatherSystem


def main() -> None:
    cfg = SimConfig(
        terrain_size=1024,
        drone_max_speed=15.0,
        drone_cruise_altitude=50.0,
        wind_drag_coef=0.4,
        wind_battery_factor=0.6,
    )
    heightmap = np.full((1024, 1024), 10.0, dtype=np.float64)
    weather = WeatherSystem(seed=42, terrain_size=1024)

    drone = Drone(
        id=0,
        position=Vec3(512.0, 60.0, 512.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        heading=0.0,
        battery=100.0,
        sensor_active=False,
        status=DroneStatus.ACTIVE,
        max_speed=15.0,
    )

    dt = 1.0 / 20.0
    total_time = 30.0
    ticks = int(total_time / dt)
    start_pos = drone.position

    print(f"Initial position: ({drone.position.x:.1f}, {drone.position.z:.1f})")
    print(f"Initial wind: speed={weather.get_wind_speed():.2f} m/s, "
          f"dir={weather.get_wind_direction():.2f} rad")

    for i in range(ticks):
        weather.update(i * dt)
        drone = update_drone_physics(
            drone, dt, heightmap, cfg, wind_fn=weather.get_wind_at,
        )

    drift_x = drone.position.x - start_pos.x
    drift_z = drone.position.z - start_pos.z
    drift = (drift_x**2 + drift_z**2) ** 0.5
    print(f"After {total_time:.0f}s: position=({drone.position.x:.1f}, {drone.position.z:.1f})")
    print(f"Net drift: {drift:.1f}m ({drift_x:+.1f} east, {drift_z:+.1f} north)")
    print(f"Final wind: speed={weather.get_wind_speed():.2f} m/s, "
          f"dir={weather.get_wind_direction():.2f} rad, gusting={weather._gusting}")

    # Sanity: drift should be nonzero and roughly consistent with wind direction.
    assert drift > 5.0, f"Expected drift > 5m, got {drift:.1f}m"
    print("\nOK — wind drift confirmed end-to-end.")


if __name__ == "__main__":
    main()
