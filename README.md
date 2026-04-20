# Autonomous Drone Swarm Coordination Simulator  

A real-time multi-agent simulation of autonomous drones performing search-and-rescue over procedurally generated terrain. Features a three-tier hybrid AI architecture, browser-based 3D visualization with fog-of-war, and human-in-the-loop interaction via natural language.

![Drone Swarm](https://img.shields.io/badge/drones-multi--agent-00c8ff) ![Python](https://img.shields.io/badge/python-3.11+-blue) ![Three.js](https://img.shields.io/badge/three.js-3D-green) ![Claude](https://img.shields.io/badge/claude-mission_planner-ff6b00) ![Mistral](https://img.shields.io/badge/mistral-drone_AI-purple)

## Quick Start

```bash
make install   # Install Python + Node dependencies
make run       # Start backend + frontend
# Open http://localhost:5173
```

## What It Does

Twenty autonomous drones deploy from a central base and search procedurally generated terrain for survivors. They coordinate through range-limited communication, adapt when drones fail, and accept human commands via natural language.

### The Simulation

- **Procedural terrain** — Multi-octave simplex noise generates unique landscapes with 6 biomes (water, beach, forest, urban, mountain, snow)
- **Realistic drone physics** — Smooth acceleration, battery drain (speed/altitude/sensor dependent), altitude maintenance above terrain
- **Sensor model** — Distance-based detection probability, range-limited field of view
- **Communication constraints** — Drones can only share information with neighbors within 100m
- **Failure modes** — Random sensor/comms failures, battery depletion, with automatic reassignment
- **Fog-of-war** — Partial observability — the map reveals only where drones have searched

### Three-Tier AI Architecture

```
┌─────────────────────────────────────────────────┐
│  Tier 1: Claude (Mission Planner)               │
│  Strategic decisions every ~10s                  │
│  Zone priorities, fleet reassignment, briefings  │
├─────────────────────────────────────────────────┤
│  Tier 2: Mistral (Drone Reasoner)               │
│  Tactical decisions every ~5s per drone          │
│  Local search/investigate/reposition decisions   │
├─────────────────────────────────────────────────┤
│  Tier 3: Classical AI (every tick, 20Hz)         │
│  A* pathfinding, lawnmower/frontier/priority     │
│  search, potential field repulsion, zone sweeps  │
└─────────────────────────────────────────────────┘
```

- **Classical AI** handles the real-time 20Hz loop — no API latency
- **LLMs** provide strategic/tactical intelligence on a slower cadence
- System works fully without LLM API keys (classical AI fallback)

### Human-in-the-Loop

- **Click** a drone to select it and see its status
- **Right-click** terrain to send a selected drone to that location
- **Natural language chat** (press T) — "Focus search on the north", "Pull back drone 7", "What's our status?"
- The system **adapts** to human overrides and records preferences

## Architecture

```
Browser (Three.js + TypeScript)
    │ WebSocket (JSON, 20Hz)
    ▼
Python Backend (FastAPI + asyncio)
    ├── Simulation Engine (20Hz fixed timestep)
    │   ├── Terrain Generator (OpenSimplex noise)
    │   ├── Drone Physics (acceleration, battery, sensors)
    │   └── Event System (discoveries, failures)
    ├── Agent Layer
    │   ├── Classical AI (A*, search patterns, coordination)
    │   ├── Claude Mission Planner (strategic, async)
    │   ├── Mistral Drone Reasoner (tactical, async)
    │   └── Chat Handler (NL command interpretation)
    └── WebSocket Server (state broadcast + commands)
```

## Controls

| Action | Input |
|--------|-------|
| Orbit camera | Mouse drag |
| Zoom | Scroll wheel |
| Select drone | Left-click drone |
| Send drone | Right-click terrain (with drone selected) |
| Pause / Resume | Space |
| Speed 1x / 2x / 5x | 1 / 2 / 3 |
| New simulation | N |
| Toggle chat | T |
| Close chat | Escape |

## Project Structure

```
agent_swarm/
├── backend/
│   ├── src/
│   │   ├── simulation/     # Core engine: physics, terrain, state, serialization
│   │   ├── agents/          # AI layer: pathfinding, search, LLM integration
│   │   │   └── prompts/     # LLM prompt templates
│   │   ├── server/          # FastAPI WebSocket server
│   │   └── terrain/         # Procedural terrain generator
│   └── tests/               # 91 tests
├── frontend/
│   └── src/
│       ├── scene/           # Three.js terrain renderer
│       ├── entities/        # Drone meshes, overlays
│       ├── fog/             # Fog-of-war DataTexture
│       ├── network/         # WebSocket client + types
│       └── ui/              # Interaction, chat panel
├── shared/
│   └── protocol.md          # WebSocket message schema
├── Makefile                  # One-command operations
└── CLAUDE.md                 # AI development context
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| 3D Visualization | Three.js | Free, performant instanced rendering |
| Frontend Build | Vite + TypeScript | Fast HMR, strict types |
| Backend | Python + FastAPI | Async WebSocket, AI/ML ecosystem |
| Simulation | NumPy | Vectorized physics, fog-of-war |
| Terrain | OpenSimplex | Reproducible procedural generation |
| Strategic AI | Claude API | Mission planning, NL interface |
| Tactical AI | Mistral API (free tier) | Per-drone reasoning |
| Pathfinding | A* + Potential Fields | Real-time, deterministic |

## Optional: LLM API Keys

The system works fully without API keys (classical AI only). To enable LLM agents:

```bash
cp .env.example backend/.env
# Edit backend/.env:
# ANTHROPIC_API_KEY=sk-ant-...
# MISTRAL_API_KEY=...
```

## Development

```bash
make test       # Run 91 backend tests
make lint       # Lint Python + TypeScript
make typecheck  # Type-check all code
make format     # Auto-format all code
make headless   # Run simulation without browser
```

## Performance

| Metric | Value |
|--------|-------|
| Simulation tick rate | 1,400+ ticks/sec (70x real-time) |
| WebSocket update size | ~4 KB per tick |
| Terrain initial payload | ~360 KB (128x128) |
| Drone fleet rendering | Single draw call (InstancedMesh) |
| Frontend frame budget | < 8ms at 60fps |

## Key Design Decisions

- **Immutable state** — Each simulation tick produces a new `WorldState`. No mutation between ticks ensures determinism and debuggability.
- **Hybrid AI** — Mirrors real robotics architectures with reactive (classical) and deliberative (LLM) layers. Demonstrates understanding of both paradigms.
- **Decentralized communication** — Drones share info only within comms range, creating emergent coordination patterns.
- **Graceful degradation** — Every LLM feature has a classical fallback. The system never breaks if an API is down.

## Research Areas Demonstrated

- Multi-agent coordination under partial observability
- Human-AI teaming with adaptive override handling
- Communication-constrained distributed systems
- Hybrid classical/neural AI architectures
- Real-time simulation with browser-based visualization

---

Built with [Claude Code](https://claude.ai/claude-code)
