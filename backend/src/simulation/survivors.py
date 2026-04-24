"""Survivor movement logic — mobile survivors wander or flee from drones.

Only mobile, undiscovered survivors move. Once a survivor is found they stay put.
Baseline behaviour is a biome-aware random walk; when an active drone comes
within `FLEE_RADIUS`, the survivor moves AWAY from the nearest drone at an
elevated speed (models a disoriented victim reacting to rotor noise).

Movement is clamped to terrain bounds and avoids WATER biome cells.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from src.simulation.types import Biome, Drone, DroneStatus, Survivor, Terrain, Vec3

# Survivor flees when any active drone enters this radius (meters).
FLEE_RADIUS: float = 80.0
# Speed multiplier applied when fleeing. ~2.5× means a 0.5 m/s wanderer
# briefly sprints to ~1.25 m/s, enough to dodge a detection corridor.
FLEE_SPEED_MULT: float = 2.5


def update_survivors(
    survivors: tuple[Survivor, ...],
    terrain: Terrain,
    dt: float,
    rng: np.random.Generator,
    *,
    drones: tuple[Drone, ...] = (),
) -> tuple[Survivor, ...]:
    """Return a new tuple of survivors with mobile ones moved by random walk.

    Args:
        survivors: Current survivor state (frozen dataclasses).
        terrain: The simulation terrain (for bounds, heightmap, biome checks).
        dt: Time delta in seconds.
        rng: Numpy random generator for reproducible movement.

    Returns:
        New tuple of Survivor objects with updated positions for mobile survivors.
    """
    result: list[Survivor] = []
    width = terrain.width
    height = terrain.height
    heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
    biome_map: np.ndarray = terrain.biome_map  # type: ignore[assignment]

    # Pre-filter to active drones — failed / recharging drones don't scare
    # survivors. Index by position for cheap nearest lookup.
    active_drone_positions: list[tuple[float, float]] = [
        (d.position.x, d.position.z)
        for d in drones
        if d.status == DroneStatus.ACTIVE
    ]

    for survivor in survivors:
        if not survivor.mobile or survivor.discovered:
            result.append(survivor)
            continue

        # Check for fleeing: nearest active drone within FLEE_RADIUS?
        flee_vec: tuple[float, float] | None = None
        if active_drone_positions:
            nearest_d_sq = FLEE_RADIUS * FLEE_RADIUS
            nearest_dx = 0.0
            nearest_dz = 0.0
            for dx_drone, dz_drone in active_drone_positions:
                rdx = survivor.position.x - dx_drone
                rdz = survivor.position.z - dz_drone
                d_sq = rdx * rdx + rdz * rdz
                if d_sq < nearest_d_sq:
                    nearest_d_sq = d_sq
                    nearest_dx = rdx
                    nearest_dz = rdz
            if nearest_d_sq < FLEE_RADIUS * FLEE_RADIUS:
                # Unit vector away from nearest drone
                mag = math.sqrt(max(nearest_d_sq, 1e-9))
                flee_vec = (nearest_dx / mag, nearest_dz / mag)

        if flee_vec is not None:
            step = survivor.speed * FLEE_SPEED_MULT * dt
            dx = flee_vec[0] * step
            dz = flee_vec[1] * step
        else:
            # Baseline random walk. Probability gate is historical; directions
            # are re-sampled each tick so the gate is effectively always true,
            # which is the intended "wander" feel.
            angle = rng.uniform(0.0, 2.0 * math.pi)
            step = survivor.speed * dt
            dx = math.cos(angle) * step
            dz = math.sin(angle) * step

        new_x = survivor.position.x + dx
        new_z = survivor.position.z + dz

        # Clamp to terrain bounds (1 .. dim-2 to stay off edges)
        new_x = max(1.0, min(float(width - 2), new_x))
        new_z = max(1.0, min(float(height - 2), new_z))

        # Check biome at new position — reject move into WATER
        grid_col = int(new_x)
        grid_row = int(new_z)
        grid_col = max(0, min(width - 1, grid_col))
        grid_row = max(0, min(height - 1, grid_row))

        if biome_map[grid_row, grid_col] == Biome.WATER.value:
            # Reject move — survivor stays in place
            result.append(survivor)
            continue

        # Update Y (altitude) from heightmap at new XZ position
        new_y = float(heightmap[grid_row, grid_col])

        new_pos = Vec3(x=new_x, y=new_y, z=new_z)
        result.append(replace(survivor, position=new_pos))

    return tuple(result)
