"""Survivor movement logic — mobile survivors wander via random walk.

Only mobile, undiscovered survivors move. Once a survivor is found they stay put.
Movement is clamped to terrain bounds and avoids WATER biome cells.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from src.simulation.types import Biome, Survivor, Terrain, Vec3


def update_survivors(
    survivors: tuple[Survivor, ...],
    terrain: Terrain,
    dt: float,
    rng: np.random.Generator,
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

    for survivor in survivors:
        if not survivor.mobile or survivor.discovered:
            result.append(survivor)
            continue

        # Probabilistic direction change (~every 3 seconds on average)
        if rng.random() < dt / 3.0:
            angle = rng.uniform(0.0, 2.0 * math.pi)
        else:
            # Keep moving in a random direction each tick for simplicity;
            # the probability gate above controls *how often* direction changes
            # matter, but we always attempt a small step.
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
