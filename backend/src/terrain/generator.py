"""Procedural terrain generator for the drone swarm search-and-rescue simulation.

Produces a heightmap from multi-octave OpenSimplex noise, classifies biomes based
on elevation and moisture, and places survivors with biome-weighted probabilities.
"""

from __future__ import annotations

import numpy as np
from opensimplex import OpenSimplex

from src.simulation.types import Biome, SimConfig, Survivor, Terrain, Vec3

# ---------------------------------------------------------------------------
# Noise parameters
# ---------------------------------------------------------------------------

_OCTAVES = 5
_LACUNARITY = 2.0  # frequency multiplier per octave
_PERSISTENCE = 0.5  # amplitude multiplier per octave
_BASE_SCALE = 0.005  # controls how "zoomed-in" the noise looks (lower = smoother)

# Moisture noise uses a coarser scale so biome regions are broader.
_MOISTURE_SCALE = 0.004
_MOISTURE_SEED_OFFSET = 1000

# ---------------------------------------------------------------------------
# Biome thresholds (applied to normalised 0-1 elevation)
# ---------------------------------------------------------------------------

_ELEV_WATER_MAX = 0.15
_ELEV_BEACH_MAX = 0.22
_ELEV_MID_MAX = 0.55  # forest / urban band
_ELEV_MOUNTAIN_MAX = 0.80
_MOISTURE_FOREST_MIN = 0.4

# Survivor spawn weights per biome (WATER and SNOW = 0 → impassable).
_BIOME_SURVIVOR_WEIGHT: dict[int, float] = {
    Biome.WATER.value: 0.0,
    Biome.BEACH.value: 1.0,
    Biome.FOREST.value: 3.0,
    Biome.URBAN.value: 5.0,
    Biome.MOUNTAIN.value: 0.5,
    Biome.SNOW.value: 0.0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_terrain(config: SimConfig) -> Terrain:
    """Generate a complete ``Terrain`` from the given simulation config.

    The returned terrain contains:
    * A heightmap (float64, values 0 … max_elevation)
    * A biome map (int32, values are ``Biome`` enum ordinals)
    * A tuple of survivors placed according to biome weights
    """
    size = config.terrain_size
    seed = config.terrain_seed

    # 1. Raw noise → normalised [0, 1]
    normalised = _generate_noise_map(size, size, seed, _BASE_SCALE)

    # 2. Scale to world elevation
    heightmap = normalised * config.max_elevation

    # 3. Moisture layer (independent noise with offset seed)
    moisture = _generate_noise_map(size, size, seed + _MOISTURE_SEED_OFFSET, _MOISTURE_SCALE)

    # 4. Biome classification (uses normalised elevation, not world-scaled)
    biome_map = _classify_biomes(normalised, moisture)

    # 5. Place survivors
    survivors = _place_survivors(
        terrain_height=size,
        terrain_width=size,
        heightmap=heightmap,
        biome_map=biome_map,
        count=config.survivor_count,
        seed=seed,
    )

    return Terrain(
        width=size,
        height=size,
        max_elevation=config.max_elevation,
        heightmap=heightmap,
        biome_map=biome_map,
        survivors=survivors,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_noise_map(
    rows: int,
    cols: int,
    seed: int,
    base_scale: float,
) -> np.ndarray:
    """Return a 2-D noise array (shape *rows* × *cols*) normalised to [0, 1].

    Uses multi-octave OpenSimplex noise for natural-looking terrain.
    """
    noise_gen = OpenSimplex(seed=seed)
    noise_map = np.zeros((rows, cols), dtype=np.float64)

    amplitude = 1.0
    frequency = base_scale
    max_possible = 0.0  # theoretical max for normalisation

    for _ in range(_OCTAVES):
        for row in range(rows):
            for col in range(cols):
                noise_map[row, col] += amplitude * noise_gen.noise2(
                    col * frequency,
                    row * frequency,
                )
        max_possible += amplitude
        amplitude *= _PERSISTENCE
        frequency *= _LACUNARITY

    # Normalise using actual data range to guarantee full [0, 1] span.
    # The theoretical range [-max_possible, max_possible] is rarely reached in practice
    # which would compress the output into a narrow band around 0.5.
    lo, hi = float(noise_map.min()), float(noise_map.max())
    if hi - lo < 1e-12:
        # Degenerate case: uniform noise — return flat 0.5
        noise_map[:] = 0.5
    else:
        noise_map = (noise_map - lo) / (hi - lo)
    return noise_map


def _classify_biomes(
    normalised_heightmap: np.ndarray,
    moisture_map: np.ndarray,
) -> np.ndarray:
    """Classify each cell into a ``Biome`` based on normalised elevation and moisture.

    Returns an int32 array of the same shape, with values matching ``Biome`` enum ints.
    """
    biome_map = np.full(normalised_heightmap.shape, Biome.FOREST.value, dtype=np.int32)

    # Apply thresholds from most extreme first so later masks can override simpler ones.
    water_mask = normalised_heightmap < _ELEV_WATER_MAX
    beach_mask = (normalised_heightmap >= _ELEV_WATER_MAX) & (
        normalised_heightmap < _ELEV_BEACH_MAX
    )
    mid_mask = (normalised_heightmap >= _ELEV_BEACH_MAX) & (normalised_heightmap < _ELEV_MID_MAX)
    mountain_mask = (normalised_heightmap >= _ELEV_MID_MAX) & (
        normalised_heightmap < _ELEV_MOUNTAIN_MAX
    )
    snow_mask = normalised_heightmap >= _ELEV_MOUNTAIN_MAX

    biome_map[water_mask] = Biome.WATER.value
    biome_map[beach_mask] = Biome.BEACH.value
    biome_map[mountain_mask] = Biome.MOUNTAIN.value
    biome_map[snow_mask] = Biome.SNOW.value

    # Mid-elevation split: forest (wet) vs urban (dry)
    forest_mask = mid_mask & (moisture_map > _MOISTURE_FOREST_MIN)
    urban_mask = mid_mask & (moisture_map <= _MOISTURE_FOREST_MIN)
    biome_map[forest_mask] = Biome.FOREST.value
    biome_map[urban_mask] = Biome.URBAN.value

    return biome_map


def _place_survivors(
    terrain_height: int,
    terrain_width: int,
    heightmap: np.ndarray,
    biome_map: np.ndarray,
    count: int,
    seed: int,
) -> tuple[Survivor, ...]:
    """Place *count* survivors on the terrain using biome-weighted random selection.

    Cells with zero weight (WATER, SNOW) will never contain a survivor.
    """
    if count <= 0:
        return ()

    rng = np.random.default_rng(seed)

    # Build a flat weight array over every grid cell.
    weight_lookup = np.zeros(len(Biome), dtype=np.float64)
    for biome_val, weight in _BIOME_SURVIVOR_WEIGHT.items():
        weight_lookup[biome_val] = weight

    # Vectorised lookup: biome_map values index directly into weight_lookup.
    flat_biome = biome_map.ravel()
    weights = weight_lookup[flat_biome]

    total_weight = weights.sum()
    if total_weight == 0.0:
        # Degenerate terrain — fall back to uniform placement on land.
        weights = np.where(flat_biome != Biome.WATER.value, 1.0, 0.0)
        total_weight = weights.sum()
        if total_weight == 0.0:
            return ()  # entire map is water

    probabilities = weights / total_weight

    # Draw unique cell indices (without replacement, up to available cells).
    eligible_count = int(np.count_nonzero(weights))
    actual_count = min(count, eligible_count)
    chosen_indices = rng.choice(
        terrain_height * terrain_width,
        size=actual_count,
        replace=False,
        p=probabilities,
    )

    flat_heightmap = heightmap.ravel()
    survivors: list[Survivor] = []
    for i, cell_idx in enumerate(chosen_indices):
        row, col = divmod(int(cell_idx), terrain_width)
        elevation = float(flat_heightmap[cell_idx])
        survivors.append(
            Survivor(
                id=i,
                position=Vec3(x=float(col), y=elevation, z=float(row)),
            )
        )

    return tuple(survivors)
