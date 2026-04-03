# Autonomous Drone Swarm Coordination Simulator

## Project Overview

A browser-based simulator of autonomous drone swarms performing search-and-rescue over procedurally generated terrain. Features multi-agent coordination under partial observability, LLM-powered mission planning, and human-in-the-loop interaction.

## Architecture

```
Browser (Three.js + Vite + TypeScript)
    │ WebSocket (JSON messages)
    ▼
Python Backend (FastAPI + asyncio)
    ├── Simulation Engine (20Hz fixed timestep)
    ├── Agent Layer (Claude mission planner + Mistral drone agents + classical AI)
    └── WebSocket Server (state broadcast + command ingestion)
```

### Module Boundaries — STRICT

- **`backend/`** — Pure Python. Never imports frontend code. Owns all simulation state.
- **`frontend/`** — Pure TypeScript/Three.js. Never runs simulation logic. Renders state received via WebSocket.
- **`shared/protocol.md`** — The WebSocket message schema. Both sides implement against this contract. Any protocol change MUST update this file first.

### Communication Contract

Backend → Frontend: JSON state snapshots every sim tick (50ms at 20Hz)
Frontend → Backend: JSON command messages (user actions, drone commands)

All messages have a `type` field. See `shared/protocol.md` for the full schema.

## How to Run

```bash
# Backend
cd backend && uv run python -m src.server.main

# Frontend (separate terminal)
cd frontend && npm run dev
```

## How to Test

```bash
# Backend tests
cd backend && uv run pytest -x -q

# Frontend tests (when added)
cd frontend && npx vitest run
```

## Global Conventions

- **Python**: Formatted with `ruff`. Type hints everywhere. Async where I/O is involved.
- **TypeScript**: Formatted with `prettier`. Strict mode enabled.
- **State is immutable between ticks**: The sim engine produces a new state snapshot each tick. Nothing mutates state outside the tick function.
- **All randomness is seeded**: Terrain generation, drone failures, survivor placement — all take a seed parameter for reproducibility.
- **IDs**: Drones are identified by integer index (0 to N-1). Survivors by integer index. Grid cells by (row, col) tuple.
- **Coordinates**: Right-handed. X = east, Y = up (altitude), Z = north. Terrain is in the XZ plane.
- **Units**: Meters for distance, seconds for time, percentage (0-100) for battery.

## File Organization

```
agent_swarm/
├── CLAUDE.md              ← You are here (root architecture)
├── general_plan.md        ← Full project plan with phases
├── backend/
│   ├── CLAUDE.md          ← Python conventions, sim engine API
│   ├── pyproject.toml
│   └── src/
│       ├── simulation/    ← Core sim loop, world state, physics
│       ├── agents/        ← Agent layer (LLM + classical)
│       ├── server/        ← FastAPI + WebSocket
│       └── terrain/       ← Procedural terrain generation
├── frontend/
│   ├── CLAUDE.md          ← Three.js patterns, rendering
│   ├── package.json
│   └── src/
│       ├── scene/         ← Three.js scene setup, camera, lighting
│       ├── entities/      ← Drone meshes, survivor markers
│       ├── ui/            ← HUD, command panel, minimap
│       ├── network/       ← WebSocket client
│       └── fog/           ← Fog-of-war rendering
└── shared/
    └── protocol.md        ← WebSocket message schema (source of truth)
```
