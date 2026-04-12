"""Search pattern generators for drone swarm exploration.

Each function computes target positions for individual drones based on their
current state and knowledge of the terrain/fog-of-war grid.  All functions are
pure — they produce a result from their inputs without side effects.

Coordinate convention:
    Grid cell (row, col) → world Vec3(col, elevation, row)
    X = east, Y = up (altitude), Z = north
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.simulation.types import Biome, Vec3

if TYPE_CHECKING:
    from src.simulation.types import Terrain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRUISE_ALTITUDE: float = 50.0

_BIOME_WEIGHTS: dict[int, float] = {
    Biome.WATER.value: 0.0,
    Biome.BEACH.value: 1.0,
    Biome.FOREST.value: 3.0,
    Biome.URBAN.value: 5.0,
    Biome.MOUNTAIN.value: 0.5,
    Biome.SNOW.value: 0.0,
}

# Pre-built relative offsets for 4-connected neighbours (row, col).
_NEIGHBOR_OFFSETS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.intp)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _grid_to_world(
    row: int,
    col: int,
    terrain: Terrain | None,
    altitude: float = _CRUISE_ALTITUDE,
) -> Vec3:
    """Convert a grid cell to a world position with the given altitude offset."""
    if terrain is not None:
        heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
        y = float(heightmap[row, col]) + altitude
    else:
        y = altitude
    return Vec3(float(col), y, float(row))


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# 1. Frontier search
# ---------------------------------------------------------------------------


def frontier_search(
    position: Vec3,
    fog_grid: np.ndarray,
    terrain: Terrain,
    survivor_positions: list[Vec3] | None = None,
) -> Vec3 | None:
    """Find the best frontier cell, biased toward survivor clusters.

    A *frontier* cell is an explored/stale cell (value 1 or 2) that has at
    least one unexplored (value 0) 4-connected neighbour.

    Candidates are scored by: ``unexplored_density * survivor_proximity_boost``
    among the top-20 closest frontiers.  This means frontiers near previous
    survivor discoveries are strongly preferred.

    Returns ``None`` when no frontier exists (i.e. the map is fully explored).
    """
    height, width = fog_grid.shape

    # --- Identify explored cells (value 1 or 2) ---
    explored_mask = fog_grid > 0

    # --- Build a mask of cells that have at least one unexplored neighbour ---
    unexplored = fog_grid == 0

    has_unexplored_neighbor = np.zeros_like(explored_mask)
    has_unexplored_neighbor[:-1, :] |= unexplored[1:, :]
    has_unexplored_neighbor[1:, :] |= unexplored[:-1, :]
    has_unexplored_neighbor[:, :-1] |= unexplored[:, 1:]
    has_unexplored_neighbor[:, 1:] |= unexplored[:, :-1]

    frontier_mask = explored_mask & has_unexplored_neighbor
    frontier_coords = np.argwhere(frontier_mask)

    if frontier_coords.shape[0] == 0:
        return None

    # --- Compute XZ distances from drone position to each frontier cell ---
    drone_col = position.x
    drone_row = position.z
    dists = np.hypot(
        frontier_coords[:, 1].astype(np.float64) - drone_col,
        frontier_coords[:, 0].astype(np.float64) - drone_row,
    )

    # --- Pick top-20 closest frontiers (increased from 10 for better survivor matching) ---
    k = min(20, frontier_coords.shape[0])
    top_indices = np.argpartition(dists, k - 1)[:k]
    candidates = frontier_coords[top_indices]

    # --- Score by unexplored density in 5×5 window ---
    unexplored_f = unexplored.astype(np.float64)
    padded = np.zeros((height + 1, width + 1), dtype=np.float64)
    padded[1:, 1:] = unexplored_f
    integral = padded.cumsum(axis=0).cumsum(axis=1)

    # --- Compute survivor proximity boost for candidates ---
    surv_boost = _survivor_proximity_boost(candidates, survivor_positions)

    best_score = -1.0
    best_row, best_col = int(candidates[0, 0]), int(candidates[0, 1])

    half = 2  # 5×5 window
    for idx in range(candidates.shape[0]):
        r, c = int(candidates[idx, 0]), int(candidates[idx, 1])
        r0 = max(r - half, 0)
        c0 = max(c - half, 0)
        r1 = min(r + half + 1, height)
        c1 = min(c + half + 1, width)
        window_sum = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        score = window_sum * surv_boost[idx]
        if score > best_score:
            best_score = score
            best_row, best_col = r, c

    heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
    y = float(heightmap[best_row, best_col])
    return Vec3(float(best_col), y, float(best_row))


# ---------------------------------------------------------------------------
# 2. Lawnmower (boustrophedon) sweep
# ---------------------------------------------------------------------------


def lawnmower_waypoints(
    zone_min: tuple[int, int],
    zone_max: tuple[int, int],
    spacing: int = 30,
    terrain: Terrain | None = None,
) -> list[Vec3]:
    """Generate a systematic back-and-forth sweep covering a rectangular zone.

    Parameters
    ----------
    zone_min:
        ``(row, col)`` of the top-left corner of the zone (inclusive).
    zone_max:
        ``(row, col)`` of the bottom-right corner of the zone (inclusive).
    spacing:
        Distance between successive sweep lines in the north (row) direction.
        Should be less than the drone sensor diameter for full coverage.
    terrain:
        If provided, Y values are taken from the heightmap plus a cruise-
        altitude offset.  Otherwise ``Y = 50``.

    Returns
    -------
    list[Vec3]
        Ordered waypoints that the drone should visit sequentially.
    """
    min_row, min_col = zone_min
    max_row, max_col = zone_max

    waypoints: list[Vec3] = []
    sweep_rows = list(range(min_row, max_row + 1, spacing))
    # Ensure the last row is included so the sweep covers the entire zone.
    if sweep_rows[-1] != max_row:
        sweep_rows.append(max_row)

    for i, row in enumerate(sweep_rows):
        if i % 2 == 0:
            # Sweep east → west (min_col → max_col)
            waypoints.append(_grid_to_world(row, min_col, terrain))
            waypoints.append(_grid_to_world(row, max_col, terrain))
        else:
            # Sweep west → east (max_col → min_col)
            waypoints.append(_grid_to_world(row, max_col, terrain))
            waypoints.append(_grid_to_world(row, min_col, terrain))

    return waypoints


# ---------------------------------------------------------------------------
# 3. Priority (biome-weighted) search
# ---------------------------------------------------------------------------


def priority_search(
    position: Vec3,
    fog_grid: np.ndarray,
    terrain: Terrain,
    survivor_positions: list[Vec3] | None = None,
) -> Vec3 | None:
    """Return the highest-priority unexplored cell as a world position.

    Priority is computed as ``biome_weight * accessibility * survivor_boost``
    where:

    - biome_weight is determined by the cell's biome (URBAN=5, FOREST=3, …).
    - accessibility is ``1 / (1 + distance_to_drone / 100)``.
    - survivor_boost rewards cells near previous survivor discoveries, since
      survivors tend to cluster (avalanches, plane crashes, hiking groups).

    Returns ``None`` when no unexplored cell has a positive score.
    """
    height, width = fog_grid.shape

    # --- Mask of unexplored cells ---
    unexplored_mask = fog_grid == 0
    unexplored_coords = np.argwhere(unexplored_mask)  # (N, 2)

    if unexplored_coords.shape[0] == 0:
        return None

    # --- Biome weights for each unexplored cell ---
    biome_map: np.ndarray = terrain.biome_map  # type: ignore[assignment]
    biome_values = biome_map[unexplored_coords[:, 0], unexplored_coords[:, 1]]

    # Vectorised lookup: default weight 0 for any unknown biome value.
    max_biome = max(_BIOME_WEIGHTS.keys())
    weight_lut = np.zeros(max_biome + 1, dtype=np.float64)
    for k, v in _BIOME_WEIGHTS.items():
        weight_lut[k] = v
    weights = weight_lut[biome_values]

    # Early exit if everything remaining is water/snow (all weights zero).
    if np.all(weights == 0.0):
        return None

    # --- Accessibility based on XZ distance to drone ---
    drone_col = position.x
    drone_row = position.z
    dists = np.hypot(
        unexplored_coords[:, 1].astype(np.float64) - drone_col,
        unexplored_coords[:, 0].astype(np.float64) - drone_row,
    )
    accessibility = 1.0 / (1.0 + dists / 100.0)

    # --- Survivor proximity boost ---
    survivor_boost = _survivor_proximity_boost(unexplored_coords, survivor_positions)

    scores = weights * accessibility * survivor_boost

    best_idx = int(np.argmax(scores))
    if scores[best_idx] <= 0.0:
        return None

    best_row = int(unexplored_coords[best_idx, 0])
    best_col = int(unexplored_coords[best_idx, 1])

    heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
    y = float(heightmap[best_row, best_col])
    return Vec3(float(best_col), y, float(best_row))


def _survivor_proximity_boost(
    cell_coords: np.ndarray,
    survivor_positions: list[Vec3] | None,
    radius: float = 500.0,
    max_boost: float = 5.0,
) -> np.ndarray:
    """Compute a multiplicative boost for cells near known survivor locations.

    Survivors tend to cluster (plane crash, avalanche, hiking group), so
    unexplored areas near previous finds should be searched first.

    Each survivor contributes a gaussian-like boost that peaks at the survivor
    location and decays to 1.0 at ``radius`` meters.  Multiple nearby survivors
    stack (additive), reflecting higher confidence in the area.

    Returns an array of shape (N,) with values >= 1.0.
    """
    n = cell_coords.shape[0]
    if not survivor_positions:
        return np.ones(n, dtype=np.float64)

    boost = np.ones(n, dtype=np.float64)
    # survivor world coords: Vec3(col, y, row) → grid (row, col) = (z, x)
    for surv in survivor_positions:
        surv_col = surv.x
        surv_row = surv.z
        dists = np.hypot(
            cell_coords[:, 1].astype(np.float64) - surv_col,
            cell_coords[:, 0].astype(np.float64) - surv_row,
        )
        # Gaussian-like falloff: peak = max_boost at distance 0, ~1.0 at radius
        # sigma chosen so exp(-(radius/sigma)^2) ≈ 0 → sigma = radius / 3
        sigma = radius / 3.0
        contribution = max_boost * np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
        boost += contribution

    return boost


# ---------------------------------------------------------------------------
# 4. Archimedean spiral waypoints
# ---------------------------------------------------------------------------


def spiral_waypoints(
    center: Vec3,
    max_radius: float = 60.0,
    spacing: float = 20.0,
    terrain: Terrain | None = None,
) -> list[Vec3]:
    """Generate an outward Archimedean spiral of waypoints from *center*.

    The spiral equation is ``r = spacing * theta / (2π)``, with a point
    sampled every 15° until ``r > max_radius``.

    Parameters
    ----------
    center:
        World-space centre of the spiral.
    max_radius:
        Maximum spiral radius in metres.
    spacing:
        Radial distance gained per full revolution.
    terrain:
        If provided, Y values come from the heightmap (+ cruise altitude)
        and points are clamped to terrain bounds.

    Returns
    -------
    list[Vec3]
        Ordered waypoints from the centre outward.
    """
    waypoints: list[Vec3] = []
    theta_step = math.radians(15.0)
    two_pi = 2.0 * math.pi
    theta = 0.0

    while True:
        r = spacing * theta / two_pi
        if r > max_radius:
            break

        # World X = east = col direction, World Z = north = row direction.
        wx = center.x + r * math.cos(theta)
        wz = center.z + r * math.sin(theta)

        if terrain is not None:
            heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
            th, tw = heightmap.shape
            col = _clamp(int(round(wx)), 0, tw - 1)
            row = _clamp(int(round(wz)), 0, th - 1)
            wy = float(heightmap[row, col]) + _CRUISE_ALTITUDE
            wx = float(col)
            wz = float(row)
        else:
            wy = _CRUISE_ALTITUDE

        waypoints.append(Vec3(wx, wy, wz))
        theta += theta_step

    return waypoints


# ---------------------------------------------------------------------------
# 5. Zone assignment for multi-drone coverage
# ---------------------------------------------------------------------------


def assign_zones(
    drone_count: int,
    terrain_width: int,
    terrain_height: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Divide the terrain into roughly equal rectangular zones.

    Uses a grid subdivision that minimises the difference between the number
    of grid cells and ``drone_count``.  When the count does not divide evenly
    the trailing zones absorb the leftover columns/rows.

    Parameters
    ----------
    drone_count:
        Number of drones (and therefore zones) to create.
    terrain_width:
        Number of columns in the terrain grid.
    terrain_height:
        Number of rows in the terrain grid.

    Returns
    -------
    list[tuple[tuple[int, int], tuple[int, int]]]
        ``[(zone_min, zone_max), ...]`` where each element is
        ``((min_row, min_col), (max_row, max_col))``.
    """
    if drone_count <= 0:
        return []

    # Find the grid subdivision (n_rows × n_cols) closest to drone_count
    # whose product is >= drone_count.
    best_rows, best_cols = 1, drone_count
    best_diff = abs(drone_count - 1 * drone_count)

    for n_rows in range(1, drone_count + 1):
        n_cols = math.ceil(drone_count / n_rows)
        diff = n_rows * n_cols - drone_count
        # Prefer splits where the aspect ratio of individual zones is closer
        # to 1 (square-ish), using diff as primary key and aspect ratio as
        # tiebreaker.
        zone_h = terrain_height / n_rows
        zone_w = terrain_width / n_cols
        aspect = max(zone_h, zone_w) / max(min(zone_h, zone_w), 1e-9)

        best_zone_h = terrain_height / best_rows
        best_zone_w = terrain_width / best_cols
        best_aspect = max(best_zone_h, best_zone_w) / max(min(best_zone_h, best_zone_w), 1e-9)

        if diff < best_diff or (diff == best_diff and aspect < best_aspect):
            best_diff = diff
            best_rows, best_cols = n_rows, n_cols

    n_rows, n_cols = best_rows, best_cols

    # Compute row and column boundaries.
    row_boundaries = _split_range(terrain_height, n_rows)
    col_boundaries = _split_range(terrain_width, n_cols)

    zones: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            if len(zones) >= drone_count:
                break
            min_row, max_row = row_boundaries[ri]
            min_col, max_col = col_boundaries[ci]

            # If this is the last zone in its row, extend to terrain edge.
            if len(zones) == drone_count - 1:
                max_row = terrain_height - 1
                max_col = terrain_width - 1

            zones.append(((min_row, min_col), (max_row, max_col)))

        if len(zones) >= drone_count:
            break

    return zones


def _split_range(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``[0, total)`` into *parts* roughly equal segments.

    Returns a list of ``(start, end)`` inclusive index pairs.
    """
    base = total // parts
    remainder = total % parts
    segments: list[tuple[int, int]] = []
    cursor = 0
    for i in range(parts):
        size = base + (1 if i < remainder else 0)
        segments.append((cursor, cursor + size - 1))
        cursor += size
    return segments
