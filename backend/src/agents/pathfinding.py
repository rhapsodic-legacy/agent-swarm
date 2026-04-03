"""A* pathfinding and drone coordination utilities for the swarm simulation.

Grid-based A* on terrain with elevation and biome costs. Includes potential-field
repulsion for inter-drone collision avoidance and path simplification for reducing
waypoint count.

Coordinates: grid (row, col) maps to world Vec3(col, heightmap[row][col], row).
Each grid cell = 1 meter.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np

from src.simulation.types import Biome, Vec3

if TYPE_CHECKING:
    from src.simulation.types import Terrain

# Pre-computed 8-directional neighbor offsets: (drow, dcol, is_diagonal)
_NEIGHBORS: list[tuple[int, int, bool]] = [
    (-1, -1, True),
    (-1, 0, False),
    (-1, 1, True),
    (0, -1, False),
    (0, 1, False),
    (1, -1, True),
    (1, 0, False),
    (1, 1, True),
]

_SQRT2 = math.sqrt(2)

# Biome movement multipliers. WATER is handled as impassable (skipped entirely).
_BIOME_MULTIPLIER: dict[int, float] = {
    Biome.WATER.value: float("inf"),
    Biome.BEACH.value: 1.0,
    Biome.FOREST.value: 1.0,
    Biome.URBAN.value: 1.0,
    Biome.MOUNTAIN.value: 2.0,
    Biome.SNOW.value: 1.5,
}


def astar_path(
    terrain: Terrain,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Find the shortest path on the terrain grid using A*.

    Parameters
    ----------
    terrain:
        Terrain object with heightmap and biome_map numpy arrays.
    start:
        Starting cell as (row, col).
    goal:
        Goal cell as (row, col).

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of (row, col) from *start* to *goal* (inclusive).
        Returns an empty list if no path exists.
    """
    heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
    biome_map: np.ndarray = terrain.biome_map  # type: ignore[assignment]
    max_elev: float = terrain.max_elevation
    rows, cols = int(terrain.height), int(terrain.width)

    sr, sc = start
    gr, gc = goal

    # Bounds check — reject obviously invalid inputs.
    if not (0 <= sr < rows and 0 <= sc < cols):
        return []
    if not (0 <= gr < rows and 0 <= gc < cols):
        return []

    # Start or goal on water is immediately unsolvable.
    if int(biome_map[sr, sc]) == Biome.WATER.value:
        return []
    if int(biome_map[gr, gc]) == Biome.WATER.value:
        return []

    # Trivial case.
    if start == goal:
        return [start]

    # Pre-fetch heightmap values for fast scalar access.
    hm = heightmap  # local alias for speed

    # Heuristic: Euclidean distance accounting for elevation difference.
    goal_h = float(hm[gr, gc])

    def _heuristic(r: int, c: int) -> float:
        dr = r - gr
        dc = c - gc
        dh = float(hm[r, c]) - goal_h
        return math.sqrt(dr * dr + dc * dc + dh * dh)

    # Priority queue entries: (f_score, tiebreaker, row, col)
    # The tiebreaker ensures FIFO ordering for equal f-scores.
    counter = 0
    open_heap: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_heap, (_heuristic(sr, sc), counter, sr, sc))
    counter += 1

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    inv_max_elev = 10.0 / max_elev if max_elev > 0 else 0.0

    while open_heap:
        _f, _tie, cr, cc = heapq.heappop(open_heap)
        current = (cr, cc)

        if current == goal:
            # Reconstruct path.
            path: list[tuple[int, int]] = [goal]
            node = goal
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        current_g = g_score[current]
        current_h = float(hm[cr, cc])

        for dr, dc, is_diag in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc

            # Bounds check.
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            neighbor = (nr, nc)
            if neighbor in closed:
                continue

            # Biome check — skip impassable cells.
            biome_val = int(biome_map[nr, nc])
            if biome_val == Biome.WATER.value:
                continue

            biome_mult = _BIOME_MULTIPLIER.get(biome_val, 1.0)

            # Cost calculation.
            base_cost = _SQRT2 if is_diag else 1.0
            neighbor_h = float(hm[nr, nc])
            elev_penalty = abs(current_h - neighbor_h) * inv_max_elev
            move_cost = base_cost * biome_mult + elev_penalty

            tentative_g = current_g + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + _heuristic(nr, nc)
                heapq.heappush(open_heap, (f, counter, nr, nc))
                counter += 1

    # No path found.
    return []


def astar_path_world(
    terrain: Terrain,
    start: Vec3,
    goal: Vec3,
) -> list[Vec3]:
    """A* pathfinding using world-space Vec3 positions.

    Converts world coordinates to grid cells, runs A*, and converts the result
    back to Vec3 positions with correct Y elevation from the heightmap.

    Parameters
    ----------
    terrain:
        Terrain object.
    start:
        World-space start position.
    goal:
        World-space goal position.

    Returns
    -------
    list[Vec3]
        World-space waypoints from start to goal with correct elevation.
        Empty list if no path exists.
    """
    heightmap: np.ndarray = terrain.heightmap  # type: ignore[assignment]
    max_row = terrain.height - 1
    max_col = terrain.width - 1

    # World Vec3(x, y, z) -> grid (row, col) where row=z, col=x.
    start_row = int(min(max(round(start.z), 0), max_row))
    start_col = int(min(max(round(start.x), 0), max_col))
    goal_row = int(min(max(round(goal.z), 0), max_row))
    goal_col = int(min(max(round(goal.x), 0), max_col))

    grid_path = astar_path(terrain, (start_row, start_col), (goal_row, goal_col))

    if not grid_path:
        return []

    # Convert grid cells back to world-space Vec3 with heightmap elevation.
    return [Vec3(float(c), float(heightmap[r, c]), float(r)) for r, c in grid_path]


def potential_field_direction(
    position: Vec3,
    other_drones: list[Vec3],
    repulsion_range: float = 30.0,
    repulsion_strength: float = 5.0,
) -> Vec3:
    """Compute a repulsion vector pushing this drone away from nearby drones.

    The repulsion operates in the XZ (horizontal) plane only. Each nearby drone
    contributes a force inversely proportional to the squared distance, directed
    away from that drone.

    Parameters
    ----------
    position:
        Current drone position in world space.
    other_drones:
        Positions of all other drones.
    repulsion_range:
        Maximum distance (meters) at which other drones exert repulsion.
    repulsion_strength:
        Maximum magnitude of the returned repulsion vector.

    Returns
    -------
    Vec3
        Horizontal repulsion vector (Y=0), magnitude clamped to *repulsion_strength*.
    """
    rx = 0.0
    rz = 0.0
    range_sq = repulsion_range * repulsion_range

    px, pz = position.x, position.z

    for other in other_drones:
        dx = px - other.x
        dz = pz - other.z
        dist_sq = dx * dx + dz * dz

        # Skip self-coincident or out-of-range drones.
        if dist_sq < 1e-12 or dist_sq > range_sq:
            continue

        # Force magnitude: inversely proportional to distance squared.
        # Direction: away from the other drone (dx, dz already point away).
        inv_dist_sq = 1.0 / dist_sq
        dist = math.sqrt(dist_sq)
        # Normalize direction, then scale by inverse-square force.
        rx += (dx / dist) * inv_dist_sq
        rz += (dz / dist) * inv_dist_sq

    # Clamp magnitude.
    mag = math.sqrt(rx * rx + rz * rz)
    if mag > repulsion_strength:
        scale = repulsion_strength / mag
        rx *= scale
        rz *= scale

    return Vec3(rx, 0.0, rz)


def simplify_path(
    path: list[tuple[int, int]],
    tolerance: int = 3,
) -> list[tuple[int, int]]:
    """Reduce waypoints by removing near-collinear points (Douglas-Peucker).

    Keeps the start, end, and any points that deviate from the straight-line
    segment by more than *tolerance* cells.

    Parameters
    ----------
    path:
        Ordered list of (row, col) grid coordinates.
    tolerance:
        Maximum perpendicular distance (in grid cells) a point may deviate
        from the line between its segment endpoints before it is kept.

    Returns
    -------
    list[tuple[int, int]]
        Simplified path preserving significant direction changes.
    """
    if len(path) <= 2:
        return list(path)

    # Iterative Douglas-Peucker to avoid deep recursion on long paths.
    # Stack entries: (start_index, end_index)
    keep: list[bool] = [False] * len(path)
    keep[0] = True
    keep[-1] = True

    stack: list[tuple[int, int]] = [(0, len(path) - 1)]
    tol_sq = float(tolerance * tolerance)

    while stack:
        si, ei = stack.pop()
        if ei - si < 2:
            continue

        # Line from path[si] to path[ei].
        sr, sc = path[si]
        er, ec = path[ei]
        dr = er - sr
        dc = ec - sc
        seg_len_sq = float(dr * dr + dc * dc)

        max_dist_sq = 0.0
        max_idx = si

        if seg_len_sq < 1e-12:
            # Degenerate segment — just measure distance from the start.
            for i in range(si + 1, ei):
                pr, pc = path[i]
                d_sq = float((pr - sr) ** 2 + (pc - sc) ** 2)
                if d_sq > max_dist_sq:
                    max_dist_sq = d_sq
                    max_idx = i
        else:
            inv_seg_len_sq = 1.0 / seg_len_sq
            for i in range(si + 1, ei):
                pr, pc = path[i]
                # Perpendicular distance squared from point to line segment.
                # Using the cross-product formulation: |cross|^2 / |seg|^2
                cross = float((pr - sr) * dc - (pc - sc) * dr)
                d_sq = cross * cross * inv_seg_len_sq
                if d_sq > max_dist_sq:
                    max_dist_sq = d_sq
                    max_idx = i

        if max_dist_sq > tol_sq:
            keep[max_idx] = True
            stack.append((si, max_idx))
            stack.append((max_idx, ei))

    return [path[i] for i in range(len(path)) if keep[i]]
