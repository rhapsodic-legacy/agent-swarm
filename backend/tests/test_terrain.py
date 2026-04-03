"""Tests for procedural terrain generation."""

from __future__ import annotations

import numpy as np

from src.simulation.types import Biome, SimConfig
from src.terrain.generator import generate_terrain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = SimConfig(
    terrain_size=64,
    terrain_seed=42,
    max_elevation=200.0,
    survivor_count=10,
    drone_count=4,
)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_terrain():
    """Two generate_terrain calls with the same config must yield bit-identical output."""
    t1 = generate_terrain(_SMALL_CFG)
    t2 = generate_terrain(_SMALL_CFG)

    np.testing.assert_array_equal(t1.heightmap, t2.heightmap)
    np.testing.assert_array_equal(t1.biome_map, t2.biome_map)
    assert len(t1.survivors) == len(t2.survivors)
    for s1, s2 in zip(t1.survivors, t2.survivors, strict=True):
        assert s1.position == s2.position
        assert s1.id == s2.id


def test_different_seed_produces_different_terrain():
    cfg2 = SimConfig(
        terrain_size=64,
        terrain_seed=99,
        max_elevation=200.0,
        survivor_count=10,
        drone_count=4,
    )
    t1 = generate_terrain(_SMALL_CFG)
    t2 = generate_terrain(cfg2)

    # Heightmaps should differ with different seeds
    assert not np.array_equal(t1.heightmap, t2.heightmap)


# ---------------------------------------------------------------------------
# Heightmap bounds
# ---------------------------------------------------------------------------


def test_heightmap_values_within_bounds():
    terrain = generate_terrain(_SMALL_CFG)
    hmap = terrain.heightmap

    assert float(np.min(hmap)) >= 0.0
    assert float(np.max(hmap)) <= terrain.max_elevation + 1e-9


def test_heightmap_uses_full_range():
    """The normalisation step should ensure values span the full range."""
    terrain = generate_terrain(_SMALL_CFG)
    hmap = terrain.heightmap

    # Minimum should be very close to 0, maximum close to max_elevation
    assert float(np.min(hmap)) < terrain.max_elevation * 0.05
    assert float(np.max(hmap)) > terrain.max_elevation * 0.95


# ---------------------------------------------------------------------------
# Biome map
# ---------------------------------------------------------------------------


def test_biome_map_contains_only_valid_values():
    terrain = generate_terrain(_SMALL_CFG)
    valid_biome_values = {b.value for b in Biome}
    unique_values = set(np.unique(terrain.biome_map).tolist())

    assert unique_values.issubset(valid_biome_values), (
        f"Unexpected biome values: {unique_values - valid_biome_values}"
    )


def test_biome_map_has_multiple_biomes():
    """A 64x64 terrain with default settings should produce several distinct biomes."""
    terrain = generate_terrain(_SMALL_CFG)
    unique_count = len(np.unique(terrain.biome_map))
    assert unique_count >= 3, f"Expected at least 3 biomes, got {unique_count}"


# ---------------------------------------------------------------------------
# Survivors
# ---------------------------------------------------------------------------


def test_survivor_count_matches_config():
    terrain = generate_terrain(_SMALL_CFG)
    assert len(terrain.survivors) == _SMALL_CFG.survivor_count


def test_survivor_positions_not_in_water():
    """Survivors should never be placed on WATER cells."""
    terrain = generate_terrain(_SMALL_CFG)

    for s in terrain.survivors:
        col = int(s.position.x)
        row = int(s.position.z)
        biome_val = int(terrain.biome_map[row, col])
        assert biome_val != Biome.WATER.value, f"Survivor {s.id} placed at ({col}, {row}) on WATER"


def test_survivor_positions_not_in_snow():
    """Survivors should never be placed on SNOW cells (weight is 0)."""
    terrain = generate_terrain(_SMALL_CFG)

    for s in terrain.survivors:
        col = int(s.position.x)
        row = int(s.position.z)
        biome_val = int(terrain.biome_map[row, col])
        assert biome_val != Biome.SNOW.value, f"Survivor {s.id} placed at ({col}, {row}) on SNOW"


def test_survivor_positions_within_terrain_bounds():
    terrain = generate_terrain(_SMALL_CFG)

    for s in terrain.survivors:
        assert 0 <= s.position.x < terrain.width, f"Survivor {s.id} x out of bounds"
        assert 0 <= s.position.z < terrain.height, f"Survivor {s.id} z out of bounds"


def test_survivor_elevation_matches_heightmap():
    """Survivor Y should equal the heightmap elevation at their XZ position."""
    terrain = generate_terrain(_SMALL_CFG)

    for s in terrain.survivors:
        col = int(s.position.x)
        row = int(s.position.z)
        expected_y = float(terrain.heightmap[row, col])
        assert abs(s.position.y - expected_y) < 1e-9, (
            f"Survivor {s.id} elevation {s.position.y} != heightmap {expected_y}"
        )


def test_zero_survivors():
    cfg = SimConfig(
        terrain_size=64,
        terrain_seed=42,
        max_elevation=200.0,
        survivor_count=0,
        drone_count=4,
    )
    terrain = generate_terrain(cfg)
    assert len(terrain.survivors) == 0


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


def test_terrain_dimensions_match_config():
    terrain = generate_terrain(_SMALL_CFG)
    hmap = terrain.heightmap
    bmap = terrain.biome_map

    assert terrain.width == _SMALL_CFG.terrain_size
    assert terrain.height == _SMALL_CFG.terrain_size
    assert hmap.shape == (_SMALL_CFG.terrain_size, _SMALL_CFG.terrain_size)
    assert bmap.shape == (_SMALL_CFG.terrain_size, _SMALL_CFG.terrain_size)


def test_terrain_non_square_equivalent_sizes():
    """terrain_size is used for both width and height in the current API."""
    for size in (32, 48):
        cfg = SimConfig(terrain_size=size, terrain_seed=7, survivor_count=3, drone_count=2)
        terrain = generate_terrain(cfg)
        assert terrain.width == size
        assert terrain.height == size
        assert terrain.heightmap.shape == (size, size)
