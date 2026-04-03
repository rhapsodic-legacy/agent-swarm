# Backend — Python Simulation Engine & Server

## Running

```bash
uv run python -m src.server.main          # Start WebSocket server (port 8765)
uv run python -m src.simulation.run --headless  # Run sim without frontend
uv run pytest -x -q                       # Run tests
```

## Python Conventions

- **Formatter/Linter**: `ruff` (configured in pyproject.toml)
- **Type hints**: Required on all function signatures. Use `from __future__ import annotations`.
- **Imports**: Use absolute imports from `src.` prefix (e.g., `from src.simulation.drone import Drone`)
- **Dataclasses**: Use `@dataclass(frozen=True)` for state objects (immutability enforced)
- **Async**: All I/O (WebSocket, LLM API calls) must be async. The sim loop itself is synchronous for determinism.
- **No global mutable state**: All state lives in the `WorldState` object passed through the sim loop.

## Simulation Engine Architecture

```
WorldState (immutable snapshot)
    │
    ▼
tick(world: WorldState, dt: float, commands: list[Command]) -> WorldState
    │
    ├── update_drone_physics(world, dt)    → new positions, velocities, battery
    ├── update_sensors(world)              → new detections
    ├── update_communications(world)       → message passing between nearby drones
    ├── update_agents(world, commands)     → agent decisions (classical + LLM)
    ├── update_fog_of_war(world)           → exploration grid update
    └── process_events(world)              → failures, discoveries
    │
    ▼
New WorldState (broadcast via WebSocket)
```

### Key Types

- `WorldState` — The complete simulation state. Terrain + all drones + all survivors + fog grid + tick count.
- `Drone` — Frozen dataclass: id, position (Vec3), velocity (Vec3), heading, battery, sensor_active, comms_active, status (active/failed/returning).
- `Terrain` — Heightmap (2D numpy array), biome map (2D int array), survivor positions.
- `Vec3` — Named tuple: (x, y, z). X=east, Y=up, Z=north.
- `Command` — Tagged union of user/agent commands: MoveTo, SearchArea, ReturnToBase, Override, etc.

### Terrain Generator

Located in `src/terrain/`. Takes a seed and config (size, roughness, biome weights). Returns a `Terrain` object.
- Heightmap: OpenSimplex noise, multi-octave, normalized to 0-1 then scaled to max elevation.
- Biomes: Derived from elevation + a moisture noise layer.
- Survivors: Poisson-disc sampling weighted by biome (more in urban/forest, fewer in water/mountain).

### Agent Layer

Located in `src/agents/`. See `src/agents/CLAUDE.md` for detailed patterns.
- Classical agents run every tick (fast, deterministic).
- LLM agents run async on a slower cadence (every N seconds or on events).
- Agent decisions are Commands fed back into the next tick.

## Test Patterns

- Tests live in `backend/tests/`, mirroring `src/` structure.
- Use `pytest` with no plugins needed initially.
- Terrain tests: verify seed reproducibility, bounds, biome distribution.
- Physics tests: verify conservation laws, boundary conditions, battery drain.
- Agent tests: verify behavior in scripted scenarios (mock LLM responses).
