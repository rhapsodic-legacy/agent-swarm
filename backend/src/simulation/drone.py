"""Drone physics model for the search-and-rescue swarm simulation.

Pure functions operating on immutable Drone state. Every function takes
a snapshot and returns a new snapshot --- no mutation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from src.simulation.types import (
    Biome,
    Drone,
    DroneStatus,
    SimConfig,
    Survivor,
    Vec3,
)

# Type alias for terrain lookup callables
HeightFn = Callable[[float, float], float]
BiomeFn = Callable[[float, float], int]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACCEL_RATE: float = 5.0  # m/s^2
_DETECTION_GUARANTEED_RATIO: float = 0.8  # detect within 80 % of sensor range

# Biome detection multipliers — how effectively drones can spot survivors in each biome.
# Multiplied against the detection range. Lower = harder to spot.
_BIOME_DETECTION: dict[int, float] = {
    Biome.WATER.value: 0.0,  # no survivors in water
    Biome.BEACH.value: 1.0,  # open sand, high visibility
    Biome.FOREST.value: 0.35,  # canopy concealment, very hard
    Biome.URBAN.value: 0.55,  # buildings block line-of-sight, moderate
    Biome.MOUNTAIN.value: 0.65,  # rocky terrain, some concealment
    Biome.SNOW.value: 0.95,  # high contrast, easy to spot
}

# Optimal altitudes per biome — flying at this altitude gives standard detection.
# Flying lower improves it (bonus up to 1.5x), flying higher degrades it.
_BIOME_OPTIMAL_ALT: dict[int, float] = {
    Biome.WATER.value: 60.0,
    Biome.BEACH.value: 60.0,  # open — standard high altitude is fine
    Biome.FOREST.value: 20.0,  # must fly low to see through canopy
    Biome.URBAN.value: 35.0,  # moderate — below rooftop level helps
    Biome.MOUNTAIN.value: 40.0,
    Biome.SNOW.value: 60.0,  # high is fine — high contrast
}


# ---------------------------------------------------------------------------
# 1. Physics update
# ---------------------------------------------------------------------------


def update_drone_physics(
    drone: Drone,
    dt: float,
    terrain_heightmap: np.ndarray | None,
    config: SimConfig,
    *,
    height_fn: HeightFn | None = None,
    world_bounds: tuple[int, int] | None = None,
) -> Drone:
    """Advance drone position/velocity by *dt* seconds.

    * Accelerates smoothly toward the target (clamped at ``_ACCEL_RATE``).
    * Maintains cruise altitude above the terrain surface.
    * Clamps position to terrain bounds.
    * Failed drones do not move.

    If *height_fn* and *world_bounds* are provided, uses them instead of
    *terrain_heightmap* for height lookups (chunked world support).
    """
    if drone.status is DroneStatus.FAILED:
        return replace(drone, velocity=Vec3(0.0, 0.0, 0.0))

    # Determine terrain size and height lookup
    if height_fn is not None and world_bounds is not None:
        terrain_w, terrain_h = world_bounds
        def _get_height(x: float, z: float) -> float:
            return height_fn(
                min(max(x, 0.0), terrain_w - 1),
                min(max(z, 0.0), terrain_h - 1),
            )
    else:
        assert terrain_heightmap is not None
        terrain_h, terrain_w = terrain_heightmap.shape
        def _get_height(x: float, z: float) -> float:
            ix = int(min(max(x, 0.0), terrain_w - 1))
            iz = int(min(max(z, 0.0), terrain_h - 1))
            return float(terrain_heightmap[iz, ix])  # type: ignore[index]

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
    terrain_elev = _get_height(drone.position.x, drone.position.z)
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
    floor_elev = _get_height(clamped_x, clamped_z)
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


def detect_survivors(
    drone: Drone,
    survivors: tuple[Survivor, ...],
    biome_map: np.ndarray | None = None,
    *,
    biome_fn: BiomeFn | None = None,
    height_fn: HeightFn | None = None,
    config: SimConfig | None = None,
) -> list[int]:
    """Return IDs of survivors newly detected by *drone* this tick.

    Detection is a multi-factor model:
    1. **Biome occlusion** — forest canopy, buildings, etc. reduce effective range.
    2. **Altitude** — flying lower improves detection in dense biomes.
    3. **Weather** — rain/fog reduces visibility globally.
    4. **Night penalty** — detection degrades at night.
    5. **Terrain LOS** — hills between drone and survivor block detection.
    6. **Transponders** — some survivors have beacons (always detected in range).

    Accepts either a numpy *biome_map* or a *biome_fn(x, z)* callable for
    chunked world support. Without either, falls back to flat detection.
    """
    if not drone.sensor_active or drone.status is DroneStatus.FAILED:
        return []

    if config is None:
        config = SimConfig()

    detected: list[int] = []
    sr = drone.sensor_range

    for s in survivors:
        if s.discovered:
            continue

        dx = drone.position.x - s.position.x
        dz = drone.position.z - s.position.z
        dist_xz = math.sqrt(dx * dx + dz * dz)

        # --- Transponder check: always detect if survivor has a beacon ---
        # Deterministic "has transponder" from survivor ID
        has_transponder = (s.id % 1000) < int(config.transponder_ratio * 1000)
        if has_transponder and dist_xz < config.transponder_range:
            detected.append(s.id)
            continue

        # --- Biome modifier ---
        biome_mod = 1.0
        optimal_alt = 50.0
        biome_val = Biome.BEACH.value  # default: open
        if biome_fn is not None:
            biome_val = biome_fn(s.position.x, s.position.z)
            biome_mod = _BIOME_DETECTION.get(biome_val, 0.5)
            optimal_alt = _BIOME_OPTIMAL_ALT.get(biome_val, 50.0)
        elif biome_map is not None:
            s_row = int(min(max(s.position.z, 0), biome_map.shape[0] - 1))
            s_col = int(min(max(s.position.x, 0), biome_map.shape[1] - 1))
            biome_val = int(biome_map[s_row, s_col])
            biome_mod = _BIOME_DETECTION.get(biome_val, 0.5)
            optimal_alt = _BIOME_OPTIMAL_ALT.get(biome_val, 50.0)

        # --- Enhanced occlusion from config ---
        if biome_val == Biome.FOREST.value:
            biome_mod *= (1.0 - config.canopy_occlusion)
        elif biome_val == Biome.URBAN.value:
            biome_mod *= (1.0 - config.urban_occlusion)

        # --- Altitude bonus ---
        drone_alt_above_terrain = max(drone.position.y - s.position.y, 1.0)
        alt_ratio = drone_alt_above_terrain / max(optimal_alt, 1.0)
        if alt_ratio <= 0.5:
            alt_bonus = 1.5
        elif alt_ratio <= 1.0:
            alt_bonus = 1.0 + 0.5 * (1.0 - alt_ratio)
        elif alt_ratio <= 2.0:
            alt_bonus = 1.0 - 0.4 * (alt_ratio - 1.0)
        else:
            alt_bonus = 0.6

        # --- Weather visibility ---
        weather_mod = config.weather_visibility

        # --- Terrain line-of-sight check ---
        los_ok = True
        if config.detection_requires_los and height_fn is not None and dist_xz > 10.0:
            # Sample terrain height at midpoint between drone and survivor
            mid_x = (drone.position.x + s.position.x) / 2.0
            mid_z = (drone.position.z + s.position.z) / 2.0
            mid_terrain_h = height_fn(mid_x, mid_z)
            # Interpolate expected altitude at midpoint
            drone_h = drone.position.y
            survivor_h = s.position.y
            expected_h = (drone_h + survivor_h) / 2.0
            if mid_terrain_h > expected_h:
                los_ok = False  # terrain blocks line of sight

        if not los_ok:
            continue

        # --- Final effective range ---
        effective_range = (
            sr * biome_mod * alt_bonus * weather_mod * _DETECTION_GUARANTEED_RATIO
        )
        if dist_xz < effective_range:
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
