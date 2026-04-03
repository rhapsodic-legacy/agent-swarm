"""Tests for A* pathfinding, potential-field repulsion, and path simplification."""

from __future__ import annotations

import math

import numpy as np

from src.agents.pathfinding import (
    astar_path,
    astar_path_world,
    potential_field_direction,
    simplify_path,
)
from src.simulation.types import Biome, Terrain, Vec3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_terrain(
    size: int = 32,
    biome: int = Biome.FOREST.value,
    heightmap: np.ndarray | None = None,
    biome_map: np.ndarray | None = None,
) -> Terrain:
    """Build a simple Terrain object for testing."""
    if heightmap is None:
        heightmap = np.zeros((size, size), dtype=np.float64)
    if biome_map is None:
        biome_map = np.full((size, size), biome, dtype=np.int32)
    return Terrain(
        width=size,
        height=size,
        max_elevation=200.0,
        heightmap=heightmap,
        biome_map=biome_map,
        seed=42,
    )


# ---------------------------------------------------------------------------
# astar_path — basic reachability
# ---------------------------------------------------------------------------


def test_astar_finds_path_on_flat_terrain():
    terrain = _make_terrain()
    path = astar_path(terrain, (0, 0), (10, 10))
    assert len(path) >= 2
    assert path[0] == (0, 0)
    assert path[-1] == (10, 10)


def test_astar_returns_empty_when_goal_is_water():
    biome_map = np.full((32, 32), Biome.FOREST.value, dtype=np.int32)
    biome_map[15, 15] = Biome.WATER.value
    terrain = _make_terrain(biome_map=biome_map)
    path = astar_path(terrain, (0, 0), (15, 15))
    assert path == []


def test_astar_path_avoids_water_cells():
    """Place a water barrier across the middle; path must go around it."""
    biome_map = np.full((32, 32), Biome.FOREST.value, dtype=np.int32)
    # Water wall from col 0..30 at row 16
    biome_map[16, 0:31] = Biome.WATER.value
    terrain = _make_terrain(biome_map=biome_map)

    path = astar_path(terrain, (0, 0), (31, 0))
    assert len(path) > 0, "A path should exist going around the water barrier"
    for r, c in path:
        assert int(biome_map[r, c]) != Biome.WATER.value, f"Path cell ({r},{c}) is water"


def test_astar_prefers_flat_terrain_over_steep_slopes():
    """A path between two points should prefer the flat route over a steep one."""
    heightmap = np.zeros((32, 32), dtype=np.float64)
    # Create a steep ridge along row 16 except at col 0-5 (flat corridor)
    heightmap[16, 6:] = 200.0  # max elevation cliff

    terrain = _make_terrain(heightmap=heightmap)

    path = astar_path(terrain, (0, 3), (31, 3))
    assert len(path) > 0

    # The path should route through the flat corridor (col <= 5)
    # rather than crossing the ridge at col > 5.  Verify no cell crosses
    # the ridge at the steep columns.
    crossed_ridge_at_steep = any(r == 16 and c > 5 for r, c in path)
    assert not crossed_ridge_at_steep, "Path should avoid the steep ridge and use the flat corridor"


# ---------------------------------------------------------------------------
# astar_path_world — Vec3 conversion
# ---------------------------------------------------------------------------


def test_astar_path_world_returns_vec3_list():
    terrain = _make_terrain()
    start = Vec3(0.0, 0.0, 0.0)
    goal = Vec3(10.0, 0.0, 10.0)
    path = astar_path_world(terrain, start, goal)

    assert len(path) >= 2
    assert isinstance(path[0], Vec3)
    # First waypoint should correspond to grid (0,0) -> world (0, elev, 0)
    assert path[0].x == 0.0
    assert path[0].z == 0.0
    # Last waypoint should correspond to grid (10,10) -> world (10, elev, 10)
    assert path[-1].x == 10.0
    assert path[-1].z == 10.0


def test_astar_path_world_elevation_from_heightmap():
    heightmap = np.zeros((32, 32), dtype=np.float64)
    heightmap[10, 10] = 42.0
    terrain = _make_terrain(heightmap=heightmap)

    start = Vec3(0.0, 0.0, 0.0)
    goal = Vec3(10.0, 0.0, 10.0)
    path = astar_path_world(terrain, start, goal)

    assert len(path) >= 2
    # The last waypoint's Y should be the heightmap value at (10, 10)
    assert path[-1].y == 42.0


# ---------------------------------------------------------------------------
# potential_field_direction
# ---------------------------------------------------------------------------


def test_potential_field_zero_when_no_drones():
    pos = Vec3(10.0, 0.0, 10.0)
    result = potential_field_direction(pos, [])
    assert result.x == 0.0
    assert result.y == 0.0
    assert result.z == 0.0


def test_potential_field_pushes_away_from_nearby_drone():
    pos = Vec3(10.0, 0.0, 10.0)
    # Place another drone directly to the east (+X)
    other = Vec3(15.0, 0.0, 10.0)
    result = potential_field_direction(pos, [other])

    # Repulsion should push west (negative X)
    assert result.x < 0.0
    # Z component should be near zero since the other drone is on the same Z
    assert abs(result.z) < 1e-6
    # Y is always zero (horizontal-only repulsion)
    assert result.y == 0.0


def test_potential_field_magnitude_is_clamped():
    pos = Vec3(10.0, 0.0, 10.0)
    # Place a drone very close to produce a large repulsion force
    other = Vec3(10.1, 0.0, 10.0)
    strength = 5.0
    result = potential_field_direction(
        pos, [other], repulsion_range=30.0, repulsion_strength=strength
    )
    mag = math.sqrt(result.x**2 + result.z**2)
    assert mag <= strength + 1e-6


def test_potential_field_ignores_distant_drones():
    pos = Vec3(10.0, 0.0, 10.0)
    # Place a drone well outside repulsion range
    far_away = Vec3(500.0, 0.0, 500.0)
    result = potential_field_direction(pos, [far_away], repulsion_range=30.0)
    assert result.x == 0.0
    assert result.z == 0.0


# ---------------------------------------------------------------------------
# simplify_path
# ---------------------------------------------------------------------------


def test_simplify_straight_line_to_start_and_end():
    """A straight diagonal path should be simplified to just start + end."""
    path = [(i, i) for i in range(20)]
    simplified = simplify_path(path, tolerance=1)
    assert simplified[0] == path[0]
    assert simplified[-1] == path[-1]
    assert len(simplified) == 2


def test_simplify_preserves_turns():
    """An L-shaped path must keep the corner point."""
    # Go right along row 0, then down along col 10
    path = [(0, c) for c in range(11)] + [(r, 10) for r in range(1, 11)]
    simplified = simplify_path(path, tolerance=1)

    assert simplified[0] == (0, 0)
    assert simplified[-1] == (10, 10)
    # The corner (0, 10) must be preserved
    assert (0, 10) in simplified
    # Should be exactly 3 points: start, corner, end
    assert len(simplified) == 3


def test_simplify_short_path_unchanged():
    path = [(0, 0), (5, 5)]
    assert simplify_path(path) == [(0, 0), (5, 5)]


def test_simplify_single_point():
    path = [(3, 7)]
    assert simplify_path(path) == [(3, 7)]
