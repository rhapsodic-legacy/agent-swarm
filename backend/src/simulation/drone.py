"""Drone physics model for the search-and-rescue swarm simulation.

Pure functions operating on immutable Drone state. Every function takes
a snapshot and returns a new snapshot --- no mutation.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from src.simulation.types import (
    Drone,
    DroneStatus,
    SimConfig,
    Survivor,
    Vec3,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACCEL_RATE: float = 5.0  # m/s^2
_DETECTION_GUARANTEED_RATIO: float = 0.8  # detect within 80 % of sensor range


# ---------------------------------------------------------------------------
# 1. Physics update
# ---------------------------------------------------------------------------


def update_drone_physics(
    drone: Drone,
    dt: float,
    terrain_heightmap: np.ndarray,
    config: SimConfig,
) -> Drone:
    """Advance drone position/velocity by *dt* seconds.

    * Accelerates smoothly toward the target (clamped at ``_ACCEL_RATE``).
    * Maintains cruise altitude above the terrain surface.
    * Clamps position to terrain bounds.
    * Failed drones do not move.
    """
    if drone.status is DroneStatus.FAILED:
        return replace(drone, velocity=Vec3(0.0, 0.0, 0.0))

    terrain_h, terrain_w = terrain_heightmap.shape

    # --- desired velocity --------------------------------------------------
    if drone.target is not None:
        diff = drone.target - drone.position
        horiz = Vec3(diff.x, 0.0, diff.z)
        horiz_dist = horiz.length_xz()

        if horiz_dist < 0.5:
            # Close enough horizontally -- stop
            desired = Vec3(0.0, 0.0, 0.0)
        else:
            direction = horiz.normalized()
            # Slow down when approaching the target (simple proportional)
            approach_speed = min(drone.max_speed, horiz_dist / 1.0)
            desired = direction * approach_speed
    else:
        desired = Vec3(0.0, 0.0, 0.0)

    # --- smooth acceleration (clamp delta-v per tick) ----------------------
    dv = desired - Vec3(drone.velocity.x, 0.0, drone.velocity.z)
    dv_mag = dv.length()
    max_dv = _ACCEL_RATE * dt
    if dv_mag > max_dv:
        dv = dv.normalized() * max_dv
    new_vx = drone.velocity.x + dv.x
    new_vz = drone.velocity.z + dv.z

    # --- altitude: cruise altitude above terrain ---------------------------
    # Sample terrain elevation at current XZ (clamp indices to valid range)
    ix = int(min(max(drone.position.x, 0.0), terrain_w - 1))
    iz = int(min(max(drone.position.z, 0.0), terrain_h - 1))
    terrain_elev = float(terrain_heightmap[iz, ix])
    target_alt = terrain_elev + config.drone_cruise_altitude

    alt_diff = target_alt - drone.position.y
    # Vertical velocity toward target altitude, clamped to max speed
    new_vy = min(max(alt_diff * 2.0, -drone.max_speed), drone.max_speed)

    new_vel = Vec3(new_vx, new_vy, new_vz)

    # --- integrate position ------------------------------------------------
    new_pos = Vec3(
        drone.position.x + new_vel.x * dt,
        drone.position.y + new_vel.y * dt,
        drone.position.z + new_vel.z * dt,
    )

    # --- clamp position within terrain bounds ------------------------------
    clamped_x = min(max(new_pos.x, 0.0), float(terrain_w - 1))
    clamped_z = min(max(new_pos.z, 0.0), float(terrain_h - 1))
    # Don't let altitude go below terrain surface
    ix2 = int(min(max(clamped_x, 0.0), terrain_w - 1))
    iz2 = int(min(max(clamped_z, 0.0), terrain_h - 1))
    floor_elev = float(terrain_heightmap[iz2, ix2])
    clamped_y = max(new_pos.y, floor_elev)
    new_pos = Vec3(clamped_x, clamped_y, clamped_z)

    # --- heading from horizontal velocity ----------------------------------
    horiz_speed = Vec3(new_vel.x, 0.0, new_vel.z).length_xz()
    # heading 0 = north (+Z), clockwise.  atan2(east, north)
    new_heading = math.atan2(new_vel.x, new_vel.z) if horiz_speed > 0.1 else drone.heading

    return replace(drone, position=new_pos, velocity=new_vel, heading=new_heading)


# ---------------------------------------------------------------------------
# 2. Battery update
# ---------------------------------------------------------------------------


def update_drone_battery(drone: Drone, dt: float, config: SimConfig) -> Drone:
    """Drain battery based on speed, altitude, and sensor usage.

    Returns a new Drone with updated battery and possibly a new status
    (RETURNING if critical, FAILED if empty).
    """
    if drone.status in (DroneStatus.FAILED, DroneStatus.RECHARGING):
        return drone

    base_drain = config.drone_battery_drain_rate  # % / s

    speed = drone.velocity.length()
    speed_mult = 1.0 + speed / drone.max_speed
    alt_mult = 1.0 + drone.position.y / 200.0

    drain = base_drain * speed_mult * alt_mult * dt

    if drone.sensor_active:
        drain *= 1.20  # +20 %

    new_battery = max(drone.battery - drain, 0.0)

    new_status = drone.status
    if new_battery <= 0.0:
        new_status = DroneStatus.FAILED
        new_battery = 0.0
    elif new_battery < config.drone_battery_critical and drone.status is DroneStatus.ACTIVE:
        new_status = DroneStatus.RETURNING

    return replace(drone, battery=new_battery, status=new_status)


# ---------------------------------------------------------------------------
# 3. Random failure check
# ---------------------------------------------------------------------------


def check_drone_failures(
    drone: Drone,
    rng: np.random.Generator,
    config: SimConfig,
) -> Drone:
    """Probabilistically knock out sensor or comms each tick."""
    if drone.status is DroneStatus.FAILED:
        return drone

    new_sensor = drone.sensor_active
    new_comms = drone.comms_active

    if drone.sensor_active and rng.random() < config.sensor_failure_prob:
        new_sensor = False

    if drone.comms_active and rng.random() < config.comms_failure_prob:
        new_comms = False

    if new_sensor is drone.sensor_active and new_comms is drone.comms_active:
        return drone  # no change, reuse same object

    return replace(drone, sensor_active=new_sensor, comms_active=new_comms)


# ---------------------------------------------------------------------------
# 4. Survivor detection
# ---------------------------------------------------------------------------


def detect_survivors(drone: Drone, survivors: tuple[Survivor, ...]) -> list[int]:
    """Return IDs of survivors newly detected by *drone* this tick.

    Detection is deterministic: guaranteed within 80 % of sensor range,
    impossible beyond that.
    """
    if not drone.sensor_active or drone.status is DroneStatus.FAILED:
        return []

    detected: list[int] = []
    sr = drone.sensor_range

    for s in survivors:
        if s.discovered:
            continue
        dx = drone.position.x - s.position.x
        dz = drone.position.z - s.position.z
        dist_xz = math.sqrt(dx * dx + dz * dz)
        if dist_xz / sr < _DETECTION_GUARANTEED_RATIO:
            detected.append(s.id)

    return detected


# ---------------------------------------------------------------------------
# 5. Communication links
# ---------------------------------------------------------------------------


def compute_comms_links(drones: tuple[Drone, ...]) -> tuple[tuple[int, int], ...]:
    """Build sorted tuple of (id_a, id_b) pairs for drones in comms range.

    Both drones must be non-FAILED and have comms_active == True.
    """
    active = [d for d in drones if d.status is not DroneStatus.FAILED and d.comms_active]
    links: list[tuple[int, int]] = []

    for i in range(len(active)):
        da = active[i]
        for j in range(i + 1, len(active)):
            db = active[j]
            dx = da.position.x - db.position.x
            dz = da.position.z - db.position.z
            dist_xz = math.sqrt(dx * dx + dz * dz)
            # Use min of the two comms ranges (conservative)
            if dist_xz < min(da.comms_range, db.comms_range):
                a, b = (da.id, db.id) if da.id < db.id else (db.id, da.id)
                links.append((a, b))

    links.sort()
    return tuple(links)


# ---------------------------------------------------------------------------
# 6. Fleet creation
# ---------------------------------------------------------------------------


def create_drone_fleet(
    count: int,
    base_position: Vec3,
    config: SimConfig,
) -> tuple[Drone, ...]:
    """Create *count* drones at *base_position* with deterministic small offsets.

    Drones are arranged in a grid-like pattern so they don't overlap.
    """
    drones: list[Drone] = []
    cols = max(int(math.ceil(math.sqrt(count))), 1)
    spacing = 2.0  # meters between drones

    for i in range(count):
        row = i // cols
        col = i % cols
        # Centre the grid around the base position
        offset_x = (col - (cols - 1) / 2.0) * spacing
        offset_z = (row - (cols - 1) / 2.0) * spacing
        pos = Vec3(
            base_position.x + offset_x,
            base_position.y,
            base_position.z + offset_z,
        )
        drones.append(
            Drone(
                id=i,
                position=pos,
                velocity=Vec3(0.0, 0.0, 0.0),
                heading=0.0,
                battery=100.0,
                sensor_active=True,
                comms_active=True,
                status=DroneStatus.ACTIVE,
                current_task="idle",
                target=None,
                max_speed=config.drone_max_speed,
                sensor_range=config.drone_sensor_range,
                comms_range=config.drone_comms_range,
            )
        )

    return tuple(drones)
