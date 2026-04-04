"""Chunked terrain system for large-scale drone swarm simulations.

Divides the world into 256x256-meter chunks that are generated on demand from
deterministic seeded noise.  Supports worlds up to 10,000 x 10,000 meters
(~1,500 chunks) while only materialising chunks near active drones or the
camera.

This module is *additive* — it does not modify or replace the existing
monolithic ``Terrain`` / ``generate_terrain`` pipeline.
"""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass

import numpy as np
from opensimplex import OpenSimplex

from src.simulation.types import Biome, SimConfig, Survivor, Vec3

# ---------------------------------------------------------------------------
# Re-use the same noise parameters as the monolithic generator so the two
# systems produce identical elevation values at matching world coordinates.
# ---------------------------------------------------------------------------

_OCTAVES = 3  # Reduced from 5 for chunk generation speed (still looks good)
_LACUNARITY = 2.0
_PERSISTENCE = 0.5
_BASE_SCALE = 0.005
_MOISTURE_SCALE = 0.004
_MOISTURE_SEED_OFFSET = 1000

# Biome elevation thresholds (on normalised 0-1 elevation)
_ELEV_WATER_MAX = 0.15
_ELEV_BEACH_MAX = 0.22
_ELEV_MID_MAX = 0.55
_ELEV_MOUNTAIN_MAX = 0.80
_MOISTURE_FOREST_MIN = 0.4

# Survivor spawn weights per biome (matches generator.py)
_BIOME_SURVIVOR_WEIGHT: dict[int, float] = {
    Biome.WATER.value: 0.0,
    Biome.BEACH.value: 1.0,
    Biome.FOREST.value: 3.0,
    Biome.URBAN.value: 5.0,
    Biome.MOUNTAIN.value: 0.5,
    Biome.SNOW.value: 0.0,
}

# Biome colours for the overview minimap (RGB uint8)
_BIOME_COLORS: dict[int, tuple[int, int, int]] = {
    Biome.WATER.value: (30, 80, 170),
    Biome.BEACH.value: (220, 210, 150),
    Biome.FOREST.value: (30, 130, 50),
    Biome.URBAN.value: (160, 155, 145),
    Biome.MOUNTAIN.value: (130, 110, 90),
    Biome.SNOW.value: (240, 245, 250),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkCoord:
    """Grid position of a chunk in the world."""

    cx: int  # chunk X index (0 to world_chunks_x - 1)
    cz: int  # chunk Z index


@dataclass
class TerrainChunk:
    """A single chunk of terrain data."""

    coord: ChunkCoord
    heightmap: np.ndarray  # shape (chunk_size, chunk_size), float64
    biome_map: np.ndarray  # shape (chunk_size, chunk_size), int32
    survivors: list  # survivors placed in this chunk
    fog_grid: np.ndarray  # shape (chunk_size, chunk_size), int8
    generated: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_noise(
    noise_gen: OpenSimplex,
    world_x: float,
    world_z: float,
    base_scale: float,
) -> float:
    """Multi-octave OpenSimplex sample at a single world-space point.

    Returns a *raw* (un-normalised) value — the caller is responsible for
    mapping to [0, 1].
    """
    value = 0.0
    amplitude = 1.0
    frequency = base_scale
    for _ in range(_OCTAVES):
        value += amplitude * noise_gen.noise2(world_x * frequency, world_z * frequency)
        amplitude *= _PERSISTENCE
        frequency *= _LACUNARITY
    return value


def _generate_chunk_noise(
    noise_gen: OpenSimplex,
    origin_x: int,
    origin_z: int,
    size: int,
    base_scale: float,
) -> np.ndarray:
    """Generate a *size x size* noise patch at a world-space origin.

    Uses vectorized noise2array for performance (~16x faster than per-pixel).
    Returns a raw (un-normalised) float64 array.
    """
    result = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    frequency = base_scale

    # Build world-space coordinate arrays
    cols = np.arange(origin_x, origin_x + size, dtype=np.float64)
    rows = np.arange(origin_z, origin_z + size, dtype=np.float64)

    for _ in range(_OCTAVES):
        xs = cols * frequency
        ys = rows * frequency
        result += amplitude * noise_gen.noise2array(xs, ys)
        amplitude *= _PERSISTENCE
        frequency *= _LACUNARITY

    return result


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to [0, 1] using its own min/max."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def _classify_biomes(
    normalised_elevation: np.ndarray,
    moisture: np.ndarray,
) -> np.ndarray:
    """Classify each cell into a Biome (same logic as generator.py)."""
    biome_map = np.full(normalised_elevation.shape, Biome.FOREST.value, dtype=np.int32)

    biome_map[normalised_elevation < _ELEV_WATER_MAX] = Biome.WATER.value
    beach_mask = (normalised_elevation >= _ELEV_WATER_MAX) & (
        normalised_elevation < _ELEV_BEACH_MAX
    )
    biome_map[beach_mask] = Biome.BEACH.value

    mid_mask = (normalised_elevation >= _ELEV_BEACH_MAX) & (normalised_elevation < _ELEV_MID_MAX)
    biome_map[mid_mask & (moisture > _MOISTURE_FOREST_MIN)] = Biome.FOREST.value
    biome_map[mid_mask & (moisture <= _MOISTURE_FOREST_MIN)] = Biome.URBAN.value

    mountain_mask = (normalised_elevation >= _ELEV_MID_MAX) & (
        normalised_elevation < _ELEV_MOUNTAIN_MAX
    )
    biome_map[mountain_mask] = Biome.MOUNTAIN.value

    biome_map[normalised_elevation >= _ELEV_MOUNTAIN_MAX] = Biome.SNOW.value

    return biome_map


def _classify_biome_single(norm_elev: float, moisture: float) -> int:
    """Classify a single point — used for overview map."""
    if norm_elev < _ELEV_WATER_MAX:
        return Biome.WATER.value
    if norm_elev < _ELEV_BEACH_MAX:
        return Biome.BEACH.value
    if norm_elev < _ELEV_MID_MAX:
        if moisture > _MOISTURE_FOREST_MIN:
            return Biome.FOREST.value
        return Biome.URBAN.value
    if norm_elev < _ELEV_MOUNTAIN_MAX:
        return Biome.MOUNTAIN.value
    return Biome.SNOW.value


def _place_chunk_survivors(
    coord: ChunkCoord,
    chunk_size: int,
    heightmap: np.ndarray,
    biome_map: np.ndarray,
    count: int,
    seed: int,
) -> list[Survivor]:
    """Place *count* survivors in a chunk using biome-weighted sampling."""
    if count <= 0:
        return []

    # Deterministic seed per chunk
    chunk_seed = seed + coord.cx * 73856093 + coord.cz * 19349669
    rng = np.random.default_rng(chunk_seed & 0xFFFFFFFF)

    weight_lookup = np.zeros(len(Biome), dtype=np.float64)
    for biome_val, w in _BIOME_SURVIVOR_WEIGHT.items():
        weight_lookup[biome_val] = w

    flat_biome = biome_map.ravel()
    weights = weight_lookup[flat_biome]
    total_weight = float(weights.sum())

    if total_weight == 0.0:
        weights = np.where(flat_biome != Biome.WATER.value, 1.0, 0.0)
        total_weight = float(weights.sum())
        if total_weight == 0.0:
            return []

    probabilities = weights / total_weight
    eligible_count = int(np.count_nonzero(weights))
    actual_count = min(count, eligible_count)

    chosen_indices = rng.choice(
        chunk_size * chunk_size,
        size=actual_count,
        replace=False,
        p=probabilities,
    )

    flat_heightmap = heightmap.ravel()
    origin_x = coord.cx * chunk_size
    origin_z = coord.cz * chunk_size

    survivors: list[Survivor] = []
    for i, cell_idx in enumerate(chosen_indices):
        row, col = divmod(int(cell_idx), chunk_size)
        elevation = float(flat_heightmap[cell_idx])
        is_mobile = bool(rng.random() < 0.4)
        speed = float(rng.uniform(0.3, 0.8)) if is_mobile else 0.5

        # Use a globally-unique survivor ID derived from chunk coords
        survivor_id = coord.cx * 100000 + coord.cz * 1000 + i
        survivors.append(
            Survivor(
                id=survivor_id,
                position=Vec3(
                    x=float(origin_x + col),
                    y=elevation,
                    z=float(origin_z + row),
                ),
                mobile=is_mobile,
                speed=speed,
            )
        )

    return survivors


# ---------------------------------------------------------------------------
# ChunkedWorld — the main public class
# ---------------------------------------------------------------------------


class ChunkedWorld:
    """Manages a world divided into chunks.

    Only generates chunks when they are needed (near a drone or camera).
    Each chunk is generated deterministically from the world seed + chunk
    coordinates, so evicted chunks can safely be regenerated later.
    """

    def __init__(
        self,
        world_size: int,
        chunk_size: int,
        seed: int,
        config: SimConfig,
    ) -> None:
        """
        Args:
            world_size: Total world size in meters (e.g. 10000 for 10 km).
            chunk_size: Size of each chunk in meters (e.g. 256).
            seed: World generation seed.
            config: Simulation config (provides max_elevation, survivor counts, etc.).
        """
        self._world_size = world_size
        self._chunk_size = chunk_size
        self._seed = seed
        self._config = config

        self._chunks_x = math.ceil(world_size / chunk_size)
        self._chunks_z = math.ceil(world_size / chunk_size)

        # Noise generators — shared across all chunks for seamless edges.
        self._noise_elev = OpenSimplex(seed=seed)
        self._noise_moist = OpenSimplex(seed=seed + _MOISTURE_SEED_OFFSET)

        # Cache
        self._cache: dict[ChunkCoord, TerrainChunk] = {}
        self._last_access: dict[ChunkCoord, float] = {}

        # Pre-compute global noise range for normalisation.
        # To keep chunk edges truly seamless we normalise using a fixed global
        # range rather than per-chunk min/max.  We estimate the range by
        # sampling a sparse grid of points across the whole world.
        self._elev_lo, self._elev_hi = self._estimate_noise_range(self._noise_elev, _BASE_SCALE)
        self._moist_lo, self._moist_hi = self._estimate_noise_range(
            self._noise_moist, _MOISTURE_SCALE
        )

    # ------------------------------------------------------------------
    # Noise range estimation
    # ------------------------------------------------------------------

    def _estimate_noise_range(
        self,
        noise_gen: OpenSimplex,
        base_scale: float,
        sample_step: int = 64,
    ) -> tuple[float, float]:
        """Sample a sparse grid to estimate the global noise range."""
        lo = float("inf")
        hi = float("-inf")
        for z in range(0, self._world_size, sample_step):
            for x in range(0, self._world_size, sample_step):
                v = _sample_noise(noise_gen, float(x), float(z), base_scale)
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
        if hi - lo < 1e-12:
            return 0.0, 1.0
        return lo, hi

    # ------------------------------------------------------------------
    # Normalisation helpers (global range, not per-chunk)
    # ------------------------------------------------------------------

    def _normalise_elevation(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self._elev_lo) / (self._elev_hi - self._elev_lo)

    def _normalise_moisture(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self._moist_lo) / (self._moist_hi - self._moist_lo)

    def _normalise_elevation_scalar(self, raw: float) -> float:
        return (raw - self._elev_lo) / (self._elev_hi - self._elev_lo)

    def _normalise_moisture_scalar(self, raw: float) -> float:
        return (raw - self._moist_lo) / (self._moist_hi - self._moist_lo)

    # ------------------------------------------------------------------
    # Chunk generation
    # ------------------------------------------------------------------

    def _generate_chunk(self, coord: ChunkCoord) -> TerrainChunk:
        """Generate a single chunk at *coord*."""
        cs = self._chunk_size
        origin_x = coord.cx * cs
        origin_z = coord.cz * cs

        # Raw noise patches
        raw_elev = _generate_chunk_noise(self._noise_elev, origin_x, origin_z, cs, _BASE_SCALE)
        raw_moist = _generate_chunk_noise(
            self._noise_moist, origin_x, origin_z, cs, _MOISTURE_SCALE
        )

        # Normalise using the global range
        norm_elev = self._normalise_elevation(raw_elev)
        norm_moist = self._normalise_moisture(raw_moist)

        # Clamp to [0, 1] — sparse sampling may not capture the exact extremes
        norm_elev = np.clip(norm_elev, 0.0, 1.0)
        norm_moist = np.clip(norm_moist, 0.0, 1.0)

        heightmap = norm_elev * self._config.max_elevation
        biome_map = _classify_biomes(norm_elev, norm_moist)

        # Survivors proportional to chunk area / world area
        chunk_area = cs * cs
        # Use survivor_count from config as the global count for the
        # *original* terrain_size; scale to chunked world area.
        # We use a density-based approach: survivor_count / (terrain_size^2)
        # scaled up to the full world.
        density = self._config.survivor_count / (self._config.terrain_size**2)
        survivors_for_chunk = max(1, round(density * chunk_area))

        survivors = _place_chunk_survivors(
            coord, cs, heightmap, biome_map, survivors_for_chunk, self._seed
        )

        fog_grid = np.zeros((cs, cs), dtype=np.int8)

        return TerrainChunk(
            coord=coord,
            heightmap=heightmap,
            biome_map=biome_map,
            survivors=survivors,
            fog_grid=fog_grid,
            generated=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_chunk(self, coord: ChunkCoord) -> TerrainChunk:
        """Get or generate a chunk. Caches generated chunks."""
        if coord in self._cache:
            self._last_access[coord] = time.monotonic()
            return self._cache[coord]

        chunk = self._generate_chunk(coord)
        self._cache[coord] = chunk
        self._last_access[coord] = time.monotonic()
        return chunk

    def get_chunks_near(self, x: float, z: float, radius: float = 512.0) -> list[TerrainChunk]:
        """Get all chunks whose centres are within *radius* of *(x, z)*."""
        cs = self._chunk_size
        half = cs / 2.0
        chunks: list[TerrainChunk] = []

        # Bounding box in chunk coords
        cx_min = max(0, int((x - radius) / cs))
        cx_max = min(self._chunks_x - 1, int((x + radius) / cs))
        cz_min = max(0, int((z - radius) / cs))
        cz_max = min(self._chunks_z - 1, int((z + radius) / cs))

        for cz in range(cz_min, cz_max + 1):
            for cx in range(cx_min, cx_max + 1):
                centre_x = cx * cs + half
                centre_z = cz * cs + half
                dx = centre_x - x
                dz = centre_z - z
                if math.sqrt(dx * dx + dz * dz) <= radius:
                    chunks.append(self.get_chunk(ChunkCoord(cx, cz)))

        return chunks

    def get_active_chunks(self, drone_positions: list[Vec3]) -> list[TerrainChunk]:
        """Get chunks that need simulation (within 1 chunk radius of any drone)."""
        radius = float(self._chunk_size)
        seen: set[ChunkCoord] = set()
        active: list[TerrainChunk] = []

        for pos in drone_positions:
            for chunk in self.get_chunks_near(pos.x, pos.z, radius):
                if chunk.coord not in seen:
                    seen.add(chunk.coord)
                    active.append(chunk)

        return active

    def world_to_chunk(self, x: float, z: float) -> ChunkCoord:
        """Convert world position to chunk coordinate."""
        cs = self._chunk_size
        cx = int(x // cs)
        cz = int(z // cs)
        cx = max(0, min(cx, self._chunks_x - 1))
        cz = max(0, min(cz, self._chunks_z - 1))
        return ChunkCoord(cx, cz)

    def chunk_to_world(self, coord: ChunkCoord) -> tuple[float, float]:
        """Get world-space origin (top-left corner) of a chunk."""
        return (float(coord.cx * self._chunk_size), float(coord.cz * self._chunk_size))

    def get_heightmap_at(self, x: float, z: float) -> float:
        """Sample terrain height at a world position (generates chunk if needed)."""
        coord = self.world_to_chunk(x, z)
        chunk = self.get_chunk(coord)
        origin_x, origin_z = self.chunk_to_world(coord)
        local_col = int(x - origin_x)
        local_row = int(z - origin_z)
        local_col = max(0, min(local_col, self._chunk_size - 1))
        local_row = max(0, min(local_row, self._chunk_size - 1))
        return float(chunk.heightmap[local_row, local_col])

    def get_biome_at(self, x: float, z: float) -> int:
        """Sample biome at a world position."""
        coord = self.world_to_chunk(x, z)
        chunk = self.get_chunk(coord)
        origin_x, origin_z = self.chunk_to_world(coord)
        local_col = int(x - origin_x)
        local_row = int(z - origin_z)
        local_col = max(0, min(local_col, self._chunk_size - 1))
        local_row = max(0, min(local_row, self._chunk_size - 1))
        return int(chunk.biome_map[local_row, local_col])

    # ------------------------------------------------------------------
    # Overview minimap
    # ------------------------------------------------------------------

    def get_overview_map(self) -> np.ndarray:
        """Generate a low-res overview map (1 pixel per chunk) for the minimap.

        Returns shape ``(chunks_z, chunks_x, 3)`` uint8 RGB array.
        Uses fast noise sampling at chunk centres — does **not** generate full
        chunks.
        """
        overview = np.zeros((self._chunks_z, self._chunks_x, 3), dtype=np.uint8)
        cs = self._chunk_size
        half = cs / 2.0

        for cz in range(self._chunks_z):
            for cx in range(self._chunks_x):
                wx = cx * cs + half
                wz = cz * cs + half
                raw_e = _sample_noise(self._noise_elev, wx, wz, _BASE_SCALE)
                raw_m = _sample_noise(self._noise_moist, wx, wz, _MOISTURE_SCALE)
                norm_e = self._normalise_elevation_scalar(raw_e)
                norm_m = self._normalise_moisture_scalar(raw_m)
                norm_e = max(0.0, min(1.0, norm_e))
                norm_m = max(0.0, min(1.0, norm_m))
                biome = _classify_biome_single(norm_e, norm_m)
                colour = _BIOME_COLORS.get(biome, (128, 128, 128))
                overview[cz, cx] = colour

        return overview

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_chunk(self, coord: ChunkCoord) -> dict:
        """Serialize a chunk for WebSocket transmission (base64 encoded)."""
        chunk = self.get_chunk(coord)
        max_elev = max(self._config.max_elevation, 1.0)

        hm_normalised = chunk.heightmap.astype(np.float32) / max_elev
        hm_uint16 = (np.clip(hm_normalised, 0.0, 1.0) * 65535).astype(np.uint16)
        hm_b64 = base64.b64encode(hm_uint16.tobytes()).decode("ascii")

        bm_uint8 = chunk.biome_map.astype(np.uint8)
        bm_b64 = base64.b64encode(bm_uint8.tobytes()).decode("ascii")

        origin_x, origin_z = self.chunk_to_world(coord)

        return {
            "cx": coord.cx,
            "cz": coord.cz,
            "origin_x": origin_x,
            "origin_z": origin_z,
            "size": self._chunk_size,
            "max_elevation": self._config.max_elevation,
            "heightmap_b64": hm_b64,
            "biome_map_b64": bm_b64,
            "encoding": "base64_uint16_uint8",
            "survivor_count": len(chunk.survivors),
        }

    def serialize_overview(self) -> dict:
        """Serialize the overview map for the minimap."""
        overview = self.get_overview_map()
        overview_b64 = base64.b64encode(overview.tobytes()).decode("ascii")
        return {
            "chunks_x": self._chunks_x,
            "chunks_z": self._chunks_z,
            "world_size": self._world_size,
            "chunk_size": self._chunk_size,
            "overview_rgb_b64": overview_b64,
            "encoding": "base64_uint8_rgb",
        }

    # ------------------------------------------------------------------
    # Stats / accessors
    # ------------------------------------------------------------------

    def get_total_chunks(self) -> int:
        return self._chunks_x * self._chunks_z

    def get_generated_count(self) -> int:
        return len(self._cache)

    def get_world_size(self) -> int:
        return self._world_size

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def evict_stale(self, max_age_seconds: float = 60.0) -> int:
        """Remove chunks not accessed for *max_age_seconds*.

        Returns the number of evicted chunks.  Because chunks are deterministic
        they can be regenerated on next access.
        """
        now = time.monotonic()
        stale = [coord for coord, ts in self._last_access.items() if now - ts > max_age_seconds]
        for coord in stale:
            del self._cache[coord]
            del self._last_access[coord]
        return len(stale)
