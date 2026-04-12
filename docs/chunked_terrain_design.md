# Chunked Terrain & Adaptive Drone Behavior — Design Document

## Problem

Current terrain is monolithic (512x512 max). This limits:
- Realism: real SAR operations cover 10-100+ km²
- Battery dynamics: drones can reach any point too easily
- Resupply: trivial to return to base when everything is close
- AI challenge: not enough terrain variety to test adaptation

## Architecture: Chunked World

### Grid of Chunks

```
World: 10,000 x 10,000 meters (10km x 10km = 100 km²)
Chunk: 1,024 x 1,024 meters each
Grid:  ~10 x 10 = ~100 chunks

Each chunk is independently generated via seeded noise.
Only chunks near active drones or the camera are fully loaded.
```

### Chunk States

```
UNGENERATED  →  chunk hasn't been created yet
GENERATED    →  heightmap + biome exist in memory
ACTIVE       →  simulation ticks process drones in this chunk
RENDERED     →  frontend has mesh for this chunk
UNLOADED     →  evicted from memory (can regenerate from seed)
```

### Chunk Generation

Each chunk is generated deterministically from:
- `world_seed + chunk_x * 73856093 + chunk_z * 19349669`
- Same OpenSimplex noise but sampled at world-space coordinates
- Ensures seamless edges between chunks (noise is continuous)
- Survivors placed per-chunk with density based on biome distribution

### Simulation: Only Active Chunks

The sim engine only ticks drones/survivors in "active" chunks:
- A chunk is ACTIVE if any drone is within 1 chunk radius of it
- Fog-of-war is tracked per-chunk (1024x1024 grid per chunk)
- Global state: drone positions, mission progress, total coverage
- Per-chunk state: heightmap, biome map, fog grid, local survivors

### WebSocket Streaming

```
Initial connect:
  → Send world metadata (size, chunk grid dimensions, base position)
  → Send minimap overview (very low-res: 1 pixel per chunk, biome dominant color)

Per tick:
  → Send drone states (global, always sent — these are small)
  → Send fog updates (only for chunks visible to the camera)

On camera move / chunk enter viewport:
  → Client requests chunk terrain data
  → Server sends chunk heightmap + biome (base64, ~2MB per chunk)

Chunk eviction:
  → Client disposes chunks far from camera
  → Server can unload chunks with no drones
```

## Adaptive Drone Behavior

### Biome-Aware Flight Parameters

Drones should dynamically adjust their behavior based on the biome they're searching:

| Biome | Altitude | Spacing | Speed | Sensor Mode | Why |
|-------|----------|---------|-------|-------------|-----|
| Forest | 20m (low) | 15m (tight) | 8 m/s (slow) | Infrared (penetrate canopy) | Trees block visual, need to fly low and slow |
| Urban | 35m (medium) | 20m | 10 m/s | Visual + thermal | Buildings create shadows, moderate altitude for coverage |
| Beach | 60m (high) | 35m (wide) | 15 m/s (fast) | Visual | Open terrain, wide sweeps efficient |
| Snow | 60m (high) | 35m (wide) | 15 m/s | Visual + contrast | High contrast, easy detection |
| Mountain | 40m | 25m | 10 m/s | Mixed | Variable terrain, moderate caution |
| Water | -- | -- | -- | Skip | No survivors in water |

### How This Changes Detection

Current: flat 40m sensor range everywhere.
New: effective range depends on altitude + biome + sensor mode.

```python
effective_range = base_range * biome_modifier * altitude_modifier
```

Where:
- `altitude_modifier`: flying lower increases detection but reduces coverage width
  - `min(1.0, altitude / optimal_altitude_for_biome)`
- `biome_modifier`: the existing biome detection table (forest=0.35, etc.)
- But now: if the drone adapts (lowers altitude in forest), the modifier improves:
  - Forest at 50m: modifier = 0.35 (trees fully block)
  - Forest at 20m: modifier = 0.65 (below canopy line, much better)
  - Forest at 10m: modifier = 0.85 (nearly at canopy level, excellent but slow)

### Adaptive Spacing

When drones enter a forest zone, the coordinator should:
1. Detect the biome of the assigned zone
2. Reduce lawnmower spacing from 30m to 15m
3. Lower target cruise altitude from 50m to 20m
4. Slow drone speed to 8 m/s
5. This means the zone takes 4x longer to search — realistic

### Cluster Behavior

In high-difficulty biomes (forest, urban), drones should:
- Reduce repulsion radius (cluster tighter)
- Use overlapping search patterns (redundant passes for hidden survivors)
- Communicate findings more frequently to nearby drones
- If one drone detects a survivor in forest, nearby drones should spiral-search the area (survivor clusters are likely)

### Implementation: BiomeFlightProfile

```python
@dataclass(frozen=True)
class BiomeFlightProfile:
    cruise_altitude: float   # meters above terrain
    search_spacing: float    # meters between sweep lines
    max_speed: float         # m/s during search
    detection_modifier: float  # multiplier on base sensor range
    repulsion_range: float   # how far drones push apart
    cluster_on_find: bool    # attract nearby drones on discovery
```

The coordinator looks up the profile for each drone's current biome
and adjusts its commands accordingly.
```

## Implementation Order

1. Adaptive drone behavior (biome flight profiles) — works on current terrain
2. Chunk data structure + generator
3. Chunk-aware simulation engine
4. WebSocket chunk streaming protocol
5. Frontend chunk renderer + RTS minimap
6. Integration + testing
