# Autonomous Drone Swarm Coordination Simulator — Master Plan

> **Goal**: A portfolio-grade simulator demonstrating multi-agent coordination, partial observability, and human-AI teaming over procedurally generated terrain, visualized in a browser-based 3D environment with fog-of-war.

> **Secondary Goal**: Demonstrate the full power of Claude Code's tooling ecosystem (skills, hooks, sub-agents, hierarchical CLAUDE.md files) as a reference implementation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Three.js)                    │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐│
│  │ 3D Scene │  │ Minimap  │  │   HUD / Command Panel  ││
│  │ Terrain  │  │ Fog-of-  │  │  Chat / Zone Draw /    ││
│  │ Drones   │  │ War View │  │  Drone Select & Cmd    ││
│  └──────────┘  └──────────┘  └────────────────────────┘│
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (real-time state + commands)
                       │
┌──────────────────────┴──────────────────────────────────┐
│              Python Backend (FastAPI)                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │              WebSocket Server                        ││
│  │         (state broadcast + command ingestion)        ││
│  └──────────────────────┬──────────────────────────────┘│
│                         │                                │
│  ┌──────────────────────┴──────────────────────────────┐│
│  │            Simulation Engine (Core Loop)             ││
│  │  ┌────────────┐  ┌───────────┐  ┌────────────────┐ ││
│  │  │  Terrain   │  │  Drone    │  │  Event System  │ ││
│  │  │  Generator │  │  Physics  │  │  (failures,    │ ││
│  │  │  (Simplex) │  │  Model    │  │   discoveries) │ ││
│  │  └────────────┘  └───────────┘  └────────────────┘ ││
│  └──────────────────────┬──────────────────────────────┘│
│                         │                                │
│  ┌──────────────────────┴──────────────────────────────┐│
│  │               Agent Layer                            ││
│  │  ┌────────────────┐  ┌────────────────────────────┐ ││
│  │  │ Mission Planner│  │   Drone Agent (per drone)  │ ││
│  │  │ (Claude API)   │  │   - Mistral for reasoning  │ ││
│  │  │ - Strategy     │  │   - A* for pathfinding     │ ││
│  │  │ - Reallocation │  │   - Local comms protocol   │ ││
│  │  │ - NL Interface │  │   - Sensor model           │ ││
│  │  └────────────────┘  └────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer              | Technology               | Why                                                      |
|--------------------|--------------------------|----------------------------------------------------------|
| **3D Visualization** | Three.js               | Free, mature, stunning visuals, massive ecosystem        |
| **Frontend Build** | Vite + TypeScript        | Fast HMR, modern bundling, type safety                   |
| **Backend**        | Python + FastAPI         | Async WebSocket support, Python ecosystem for AI/sim     |
| **Simulation**     | NumPy + custom engine    | Fast vectorized physics on CPU                           |
| **Terrain**        | OpenSimplex / noise      | Procedural generation with reproducible seeds            |
| **High-Level AI**  | Claude API (Anthropic)   | Mission planning, NL interface, strategic reasoning      |
| **Drone-Level AI** | Mistral API (free tier)  | Per-drone tactical decisions, cost-effective              |
| **Pathfinding**    | A* / Potential Fields    | Real-time, deterministic, no API latency                 |
| **Communication**  | WebSocket (FastAPI)      | Real-time bidirectional state sync                       |
| **Task Queue**     | asyncio                  | Lightweight async for LLM calls without blocking sim     |

---

## Phase Breakdown

### Phase 0: Project Scaffolding & Claude Code Tooling
**Goal**: Set up the monorepo, Claude Code infrastructure, and development workflow so every subsequent phase is efficient.

- [ ] Initialize git repo with `.gitignore`
- [ ] Create monorepo structure:
  ```
  agent_swarm/
  ├── CLAUDE.md                    # Root: architecture, conventions, how modules connect
  ├── general_plan.md              # This file
  ├── backend/
  │   ├── CLAUDE.md                # Python conventions, sim engine API, how to run
  │   ├── pyproject.toml
  │   ├── src/
  │   │   ├── simulation/          # Core sim engine
  │   │   ├── agents/              # Agent layer
  │   │   ├── server/              # FastAPI WebSocket server
  │   │   └── terrain/             # Terrain generation
  │   └── tests/
  ├── frontend/
  │   ├── CLAUDE.md                # Three.js patterns, WebSocket protocol, build
  │   ├── package.json
  │   ├── src/
  │   │   ├── scene/               # Three.js scene setup
  │   │   ├── entities/            # Drone meshes, terrain mesh
  │   │   ├── ui/                  # HUD, command panel, chat
  │   │   ├── network/             # WebSocket client
  │   │   └── fog/                 # Fog-of-war rendering
  │   └── index.html
  └── shared/
      └── protocol.md              # WebSocket message schema documentation
  ```
- [ ] Create hierarchical `CLAUDE.md` files
- [ ] Configure Claude Code skills:
  - `/test` — run pytest for backend, vitest for frontend
  - `/lint` — run ruff (Python) + eslint (TS)
  - `/build` — build frontend for production
  - `/run` — start both backend and frontend dev servers
  - `/sim` — run headless simulation (no frontend)
- [ ] Configure Claude Code hooks:
  - **Pre-commit**: ruff check + ruff format --check + tsc --noEmit
  - **Post-edit (Python)**: ruff format on changed file
  - **Post-edit (TypeScript)**: prettier on changed file
- [ ] Set up Python environment (pyproject.toml, uv/pip)
- [ ] Set up frontend (Vite + Three.js + TypeScript)
- [ ] Verify both run independently

---

### Phase 1: Core Simulation Engine (Python)
**Goal**: A working simulation loop with terrain, drones, and physics — no AI, no visualization yet.

#### 1A: Terrain Generation
- [ ] Procedural heightmap using OpenSimplex noise (multi-octave)
- [ ] Biome classification based on elevation + moisture:
  - Water, beach, forest, urban ruins, mountain, snow
- [ ] Survivor placement algorithm (weighted by biome — more in urban/forest)
- [ ] Terrain serialization (heightmap + biome map + survivor positions as JSON)
- [ ] Seed-based reproducibility

#### 1B: Drone Physics Model
- [ ] Drone state: position (3D), velocity, heading, battery, sensor status, comms status
- [ ] Flight dynamics: acceleration, max speed, turn rate, altitude constraints
- [ ] Battery model: drain rate varies by speed, altitude, wind, sensor use
- [ ] Sensor model: camera FOV cone, detection probability (distance-based falloff)
- [ ] Communication model: range-limited, drones only share info with neighbors within radius
- [ ] Failure modes: battery death, random sensor failure, comms dropout

#### 1C: Simulation Loop
- [ ] Fixed-timestep simulation (configurable tick rate, default 20 Hz)
- [ ] World state manager (all drones, terrain, discovered survivors, events)
- [ ] Event system: survivor discovered, drone failure, battery critical, comms lost
- [ ] Fog-of-war as shared exploration grid (each cell: unexplored / explored / stale)
- [ ] State snapshot serialization for WebSocket broadcast
- [ ] Headless mode (run sim without frontend for testing/benchmarking)

---

### Phase 2: 3D Visualization (Three.js)
**Goal**: Render the simulation beautifully in a browser. No interactivity beyond camera yet.

#### 2A: Terrain Rendering
- [ ] Heightmap → Three.js PlaneGeometry with vertex displacement
- [ ] Biome-based color/texture mapping (vertex colors or shader)
- [ ] Directional lighting + ambient light for depth
- [ ] Water plane with transparency/reflection
- [ ] Level-of-detail (LOD) for performance if terrain is large

#### 2B: Drone Rendering
- [ ] Instanced meshes for drone fleet (scale to 50-100 drones)
- [ ] Drone model: simple geometric (octahedron or low-poly quad-rotor)
- [ ] Color-coded by status: active (green), low battery (yellow), failed (red)
- [ ] Motion trails (fading line behind each drone)
- [ ] Sensor cone visualization (semi-transparent cone below drone)
- [ ] Communication links (faint lines between drones in comms range)

#### 2C: Fog-of-War
- [ ] Overlay texture on terrain: black (unexplored) → semi-transparent (stale) → clear (explored)
- [ ] Smooth transitions as drones explore
- [ ] Minimap with fog-of-war state

#### 2D: Camera & Scene
- [ ] Orbit controls (rotate, zoom, pan)
- [ ] Follow-drone mode (camera locks to selected drone)
- [ ] Overview mode (top-down strategic view)
- [ ] Post-processing: bloom on drone lights, ambient occlusion on terrain
- [ ] Sky/environment (gradient sky or skybox)

#### 2E: WebSocket Integration
- [ ] Connect to backend WebSocket
- [ ] Receive state snapshots, interpolate between ticks for smooth rendering
- [ ] Display connection status
- [ ] Handle reconnection gracefully

---

### Phase 3: Classical AI Agents (No LLM Yet)
**Goal**: Drones autonomously search the terrain using classical algorithms. This is the baseline behavior.

#### 3A: Pathfinding
- [ ] A* on terrain grid (cost based on elevation change, biome traversability)
- [ ] Potential field avoidance (drones repel each other to spread out)
- [ ] Altitude-aware pathfinding (drones fly over obstacles)

#### 3B: Search Patterns
- [ ] Lawnmower (systematic parallel sweeps)
- [ ] Spiral (outward from a point)
- [ ] Frontier-based exploration (go to nearest unexplored boundary)
- [ ] Priority-based (search high-probability areas first)

#### 3C: Multi-Agent Coordination
- [ ] Decentralized task allocation (auction-based or market-based)
- [ ] Each drone maintains local belief map (what it knows + what neighbors shared)
- [ ] Consensus protocol for sharing discovered survivors
- [ ] Coverage optimization (minimize overlap, maximize explored area)
- [ ] Dynamic reallocation when a drone fails (neighbors absorb its assigned area)

#### 3D: Communication Protocol
- [ ] Range-limited message passing (only drones within X meters can talk)
- [ ] Message types: position update, survivor found, area claimed, help request
- [ ] Relay capability (drone can forward messages to extend range)
- [ ] Communication graph visualization in frontend

---

### Phase 4: LLM-Powered Agent Layer
**Goal**: Add intelligent high-level planning and per-drone reasoning.

#### 4A: Mistral Integration (Drone-Level)
- [ ] Each drone has a Mistral-powered reasoning module
- [ ] Inputs: local sensor data, nearby drone positions, belief map excerpt, battery state
- [ ] Outputs: tactical decision (where to search next, whether to return to base, relay priority)
- [ ] Async calls — don't block simulation; drones use classical AI between LLM decisions
- [ ] Rate limiting and batching to stay within free tier
- [ ] Fallback to pure classical AI if API is unavailable

#### 4B: Claude Integration (Mission Planner)
- [ ] Mission-level strategic planner
- [ ] Inputs: global exploration progress, drone fleet status, survivor count, terrain summary
- [ ] Outputs: zone assignments, priority shifts, fleet-wide directives
- [ ] Triggered on significant events: new survivor found, drone lost, coverage milestone
- [ ] Maintains mission log (natural language summary of decisions and reasoning)
- [ ] Generates mission briefings and status reports

#### 4C: Inter-Agent Communication via LLM
- [ ] Drones can "discuss" with nearby drones via Mistral (emergent coordination)
- [ ] Mission planner can broadcast directives that drones interpret
- [ ] All LLM interactions logged and viewable in frontend debug panel

---

### Phase 5: Human-in-the-Loop Interface
**Goal**: Let a human operator interact with the swarm at multiple levels of abstraction.

#### 5A: Direct Drone Control
- [ ] Click drone → select → issue waypoint command
- [ ] Drag to set patrol route
- [ ] Override: "return to base", "hold position", "investigate area"

#### 5B: Zone-Based Commands
- [ ] Draw rectangles/polygons on terrain to define search zones
- [ ] Assign priority levels to zones
- [ ] "Clear this area" / "Avoid this area" directives

#### 5C: Natural Language Commands (Claude-Powered)
- [ ] Chat interface in command panel
- [ ] Examples:
  - "Focus search on the urban area to the north"
  - "Pull back drone 7, its battery is too low"
  - "Why haven't we searched the eastern quadrant?"
  - "Give me a status report"
- [ ] Claude interprets commands and translates to simulation actions
- [ ] Claude explains its reasoning back to the human

#### 5D: Adaptive System
- [ ] When human overrides an agent decision, the system adapts:
  - Records the override as a preference
  - Adjusts future planning to account for human priorities
  - Can ask for clarification: "You redirected drones away from sector 4. Should I deprioritize that area?"
- [ ] Trust/autonomy slider: fully autonomous ↔ human approves all decisions

---

### Phase 6: Advanced Simulation Features
**Goal**: Increase realism and complexity for portfolio depth.

- [ ] Dynamic weather (wind fields affecting drone movement and battery drain)
- [ ] Day/night cycle affecting sensor effectiveness
- [ ] Survivor movement (some survivors wander, making them harder to find)
- [ ] Multi-objective scenarios (search-and-rescue + hazardous area mapping)
- [ ] Drone resupply mechanic (return to base, recharge, redeploy)
- [ ] Terrain hazards (no-fly zones, signal jammers)
- [ ] Replay system (record and replay missions)
- [ ] Performance metrics dashboard (coverage %, time-to-find, efficiency scores)

---

### Phase 7: Polish & Portfolio Presentation
**Goal**: Make this irresistible on a GitHub profile.

- [ ] README.md with architecture diagram, GIFs, and feature list
- [ ] Recorded demo video (2-3 minutes) showing:
  - Terrain generation
  - Swarm launch and autonomous search
  - Fog-of-war clearing in real-time
  - Drone failure and dynamic reallocation
  - Human issuing natural language commands
  - System adapting to human input
- [ ] Configuration presets (small/medium/large scenario)
- [ ] One-command startup (`make run` or similar)
- [ ] Performance profiling and optimization pass
- [ ] Code documentation for key algorithms
- [ ] Blog post / writeup explaining the architecture decisions

---

## Claude Code Tooling Strategy

### Hierarchical CLAUDE.md Files

| File | Purpose |
|------|---------|
| `/CLAUDE.md` | Project overview, architecture, module boundaries, how to run, global conventions |
| `/backend/CLAUDE.md` | Python style (ruff config), simulation engine API, agent layer patterns, test patterns |
| `/frontend/CLAUDE.md` | Three.js conventions, shader patterns, WebSocket protocol, component structure |
| `/backend/src/agents/CLAUDE.md` | Agent architecture, LLM integration patterns, prompt templates, fallback behavior |

### Skills

| Skill | Command | Purpose |
|-------|---------|---------|
| `/test` | `cd backend && pytest -x -q && cd ../frontend && npx vitest run` | Run full test suite |
| `/lint` | `cd backend && ruff check . && cd ../frontend && npx eslint src/` | Lint everything |
| `/build` | `cd frontend && npm run build` | Production build of frontend |
| `/run` | Start backend + frontend dev servers | Launch full system |
| `/sim` | `cd backend && python -m src.simulation.run --headless` | Run headless simulation |
| `/typecheck` | `cd backend && pyright && cd ../frontend && npx tsc --noEmit` | Type check everything |

### Hooks

| Trigger | Action | Purpose |
|---------|--------|---------|
| Pre-commit | `ruff check --fix`, `ruff format`, `tsc --noEmit` | Catch issues before commit |
| Post-edit `.py` | `ruff format <file>` | Auto-format Python on save |
| Post-edit `.ts` | `prettier --write <file>` | Auto-format TypeScript on save |

### Sub-Agent Strategy

For parallel development across phases, use Claude Code sub-agents:

- **Backend Agent**: Simulation engine, physics, terrain generation
- **Frontend Agent**: Three.js scene, rendering, UI components
- **Agent Layer Agent**: LLM integration, multi-agent coordination
- **Test Agent**: Write tests for completed modules
- **Integration Agent**: WebSocket protocol, state synchronization

These run in isolated worktrees to avoid conflicts, with merge points at phase boundaries.

---

## Key Design Decisions

### Why Hybrid AI (Classical + LLM)?
- Classical AI handles the 20Hz real-time loop — no API latency
- LLMs handle strategic decisions (triggered every few seconds or on events)
- This mirrors real robotics architectures (reactive layer + deliberative layer)
- Portfolio impact: shows understanding of both paradigms

### Why Decentralized Communication?
- More realistic (real drones have limited comms range)
- Creates emergent behavior (information propagation, relay chains)
- Harder problem = more impressive solution
- Visually interesting (you can see the communication graph)

### Why Fog-of-War?
- Makes partial observability tangible and visual
- Creates meaningful exploration dynamics
- The human can see what the swarm doesn't know yet
- Dramatically more engaging than omniscient view

### Why Modular Human-in-the-Loop?
- Multiple interaction modalities show design thinking
- Natural language via Claude is the "wow factor"
- Override + adaptation demonstrates human-AI teaming
- Trust slider is a research-relevant concept

---

## MVP Definition (End of Phase 2 + basic Phase 3)

The minimum impressive demo:
1. Procedural terrain renders in 3D in the browser
2. 10-20 drones launch and begin searching with classical AI
3. Fog-of-war clears as drones explore
4. Survivors are found and marked
5. Camera controls let you observe from any angle
6. Real-time WebSocket state sync

**This is the target for the first working vertical slice.**

---

## Estimated Iteration Flow

```
Phase 0 (scaffolding)  →  can develop efficiently
Phase 1 (sim engine)   →  can run headless tests
Phase 2 (visualization)→  MVP — first visual demo
Phase 3 (classical AI) →  impressive autonomous behavior
Phase 4 (LLM agents)   →  "wow" factor
Phase 5 (human-in-loop)→  full feature set
Phase 6 (advanced)     →  depth and realism
Phase 7 (polish)       →  portfolio-ready
```

Each phase produces a working, demoable state. Never a broken build.
