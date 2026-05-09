# Autonomous Drone Swarm Coordination Simulator                 

A real-time, browser-visualized simulation of an autonomous drone swarm performing search-and-rescue over procedurally generated terrain — with a unified priority-market substrate that fuses operator intent, LLM strategy, and environmental hazards into one signal the swarm bids on every tick.

[![CI](https://github.com/rhapsodic-legacy/agent-swarm/actions/workflows/ci.yml/badge.svg)](https://github.com/rhapsodic-legacy/agent-swarm/actions/workflows/ci.yml) ![Agents](https://img.shields.io/badge/agents-hybrid_classical+LLM-00c8ff) ![Python](https://img.shields.io/badge/python-3.12-blue) ![Three.js](https://img.shields.io/badge/three.js-WebGL-green) ![Claude](https://img.shields.io/badge/claude-mission_planner-ff6b00) ![Mistral](https://img.shields.io/badge/mistral-drone_reasoner-purple) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## What This Demonstrates

- **Multi-agent coordination under partial observability** — fog-of-war, range-limited comms, per-drone belief maps
- **Human-AI teaming** — natural-language operator commands, zone painting, adaptive weights, trust slider
- **Hybrid AI architecture** — classical reactive layer (20Hz) + LLM deliberative layer (async, event-driven)
- **A unified signal substrate** — every priority input (operator, LLM, environment, past outcomes) flows through one auction. Adding a new input source is a one-line producer; drones never change.
- **Reproducibility** — fully seeded terrain, survivor placement, weather, and gust cycles

## Quick Start

```bash
make install        # Python + Node dependencies
make install-hooks  # Optional: enable pre-commit checks (ruff + tsc + vitest)
make run            # Backend :8765 + frontend :5173
# Open http://localhost:5173
```

The system works **fully without API keys** — classical AI handles everything. Drop Anthropic / Mistral keys into `backend/.env` to light up the LLM tiers.

---

## The Architectural Story — The Priority Market

Every priority signal in the swarm — PoC hotspots, operator-painted zones, LLM intel pins, recent survivor finds, environmental hazards — is a `PriorityAsset`. Drones bid on assets through **one** `bid()` function. A market-clearing step resolves all assignments globally each tick.

```
┌──────────────────────────────────────────────────────────────────┐
│  Producers (compose freely — add one, swarm rebalances)          │
├──────────────────────────────────────────────────────────────────┤
│  Bayesian PoC grid    → PriorityAsset(source="poc_field")        │
│  Operator zone (paint)→ PriorityAsset(source="operator_high_zone")│
│  LLM / chat intel pin → PriorityAsset(source="intel_pin")        │
│  Survivor discovery   → PriorityAsset(source="survivor_find")    │
│  Avoid signals        → is_in_avoid_zone(x, z) callback          │
│   ├─ operator avoid zone                                          │
│   └─ active wind gust region                                     │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────┐
            │ clear_market() — global auction     │
            │                                      │
            │ bid = value × source_scale           │
            │      / (1 + dist × penalty)          │
            │      / (1 + switching_cost[task])    │
            │                                      │
            │ Filters: battery-infeasible → 0      │
            │          in avoid region  → skipped  │
            │ Capacity: per-asset saturation       │
            └─────────────────────────────────────┘
                             │
                             ▼
               Per-drone assignment this tick
```

**Why this matters**: the "incorruptible baseline" — no matter how many input modalities ship (voice, partner-swarm, sensor fusion, regulatory no-fly feeds), they all compose into the same pipeline without touching drone logic. This mirrors real-robotics substrate design.

**Read the code**: [`backend/src/agents/priority_market.py`](backend/src/agents/priority_market.py)
**Read the writeup**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — why this specific shape, what it cost, what it bought

### Adaptive weights + trust slider

The market's per-source scales aren't hardcoded. An [`AdaptiveWeights`](backend/src/agents/adaptive_weights.py) layer learns from outcomes: operator assets that produced finds get up-weighted; unused assets decay. A top-right **trust slider** (0.0–3.0) multiplies operator-origin sources at runtime — the operator can dial autonomy up or down live.

---

## Three-Tier Hybrid AI

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1 — Mission Planner (Claude)                           │
│  Trigger: event-driven (find, loss, phase transition)        │
│  Scope:   fleet-wide strategy, briefings, zone priorities    │
│  File:    backend/src/agents/mission_planner.py              │
├─────────────────────────────────────────────────────────────┤
│  Tier 2 — Drone Reasoner (Mistral)                           │
│  Trigger: every ~5s per drone, async, rate-limited           │
│  Scope:   tactical — investigate / reposition / local search │
│  File:    backend/src/agents/drone_reasoner.py               │
├─────────────────────────────────────────────────────────────┤
│  Tier 3 — Classical Controller (every tick, 20Hz)            │
│  Trigger: synchronous tick loop                              │
│  Scope:   A* pathfinding, lawnmower/frontier/priority search,│
│           potential-field repulsion, market bidding          │
│  File:    backend/src/agents/coordinator.py                  │
└─────────────────────────────────────────────────────────────┘
```

- Classical AI handles the real-time loop — **zero API latency** on the critical path
- LLMs are always in fallback mode: if an API fails, the classical tier carries the mission
- Every LLM prompt lives in [`backend/src/agents/prompts/`](backend/src/agents/prompts/) as a Jinja2 template — no inline strings

---

## Human-in-the-Loop

| Input modality | How it flows | File |
|---|---|---|
| Click a drone → right-click terrain | Direct `move_to` command, overrides agent within the tick | [`interaction.ts`](frontend/src/ui/interaction.ts) |
| Paint a zone (press `Z`) | `set_priority` → `PriorityZone` → market source scale / filter | [`zoneTool.ts`](frontend/src/ui/zoneTool.ts) |
| Chat (press `T`) | Claude parses → emits `set_intel_pin` / `zone_command` / `dismiss_intel_pin` with real coords | [`chat_handler.py`](backend/src/agents/chat_handler.py) |
| Trust slider (top-right) | Live multiplier on operator-origin priority sources | [`trustPanel.ts`](frontend/src/ui/trustPanel.ts) |

The system **records operator overrides** (`AdaptiveWeights.record_operator_override`) and bumps switching costs on that task type so the swarm learns what the operator actually cares about.

---

## Dynamic Environment

**Wind affects physics.** `WeatherSystem` produces a live wind vector at each (x, z). `update_drone_physics` adds net displacement scaled by `wind_drag_coef`; `update_drone_battery` multiplies drain for drones flying into headwinds. ([`drone.py`](backend/src/simulation/drone.py))

**Gust regions feed the market.** 6 discrete `GustRegion`s (seeded deterministically) cycle on/off on independent 60–150s sine periods. Active regions act as environmental avoid zones: the market skips PoC and intel assets inside them; in-flight drones have their targets scrubbed and re-bid. ([`weather.py`](backend/src/simulation/weather.py))

```
weather.is_hazardous_at → composes with operator avoid zones → one callback in clear_market
```

This is the substrate story made literal: an environmental signal becomes an operator-equivalent avoid zone with no special-case code in the drones.

**Day/night cycle** ([`daycycle.py`](backend/src/simulation/daycycle.py)) modulates sensor effectiveness and sun lighting. **Static hazards** ([`hazards.py`](backend/src/simulation/hazards.py)) — no-fly zones and signal jammers — layer in similarly.

---

## Simulation Features

- **Procedural terrain** — Multi-octave OpenSimplex with 6 biomes (water, beach, forest, urban, mountain, snow). Chunked world system scales to 10km × 10km at 1024m chunks ([`terrain/chunked.py`](backend/src/terrain/chunked.py))
- **Realistic drone physics** — acceleration clamping, cruise-altitude tracking, wind drag, sensor/comms/battery failure modes
- **Multi-factor detection** — biome occlusion × altitude curve × weather visibility × terrain line-of-sight × optional transponder beacons
- **Bayesian search map** — Probability-of-Containment (PoC) grid updated on every failed scan and re-focused on evidence discoveries ([`search_map.py`](backend/src/simulation/search_map.py))
- **Evidence trails** — footprints, debris, signal fires; each has a distinct posterior-update geometry (cone / ring / circle)
- **Mobile survivors** — ~40% wander with biome-aware random walk until discovered
- **5 missions** — `aircraft_crash`, `lost_hiker`, `maritime_sar`, `avalanche`, `disaster_response` — each seeds a different mission-specific prior ([`mission.py`](backend/src/simulation/mission.py))
- **Replay system** — full recording + replay for retrospective analysis ([`replay.py`](backend/src/simulation/replay.py))
- **Metrics dashboard** — MTTD, survival window %, entropy drop %, coverage rate, efficiency score ([`metrics.py`](backend/src/simulation/metrics.py))

---

## System Architecture

```
Browser (Three.js + TypeScript, strict mode)
    │  WebSocket (JSON, 10–20Hz)
    ▼
Python Backend (FastAPI + asyncio)
    ├── Simulation Engine (fixed timestep, immutable state)
    │   ├── Terrain (procedural, chunked)
    │   ├── Drone physics (wind-aware)
    │   ├── Detection (multi-factor)
    │   ├── Weather / daycycle / hazards
    │   └── Bayesian search map (PoC)
    ├── Agent Layer
    │   ├── Classical: A*, search patterns, coordinator, priority market
    │   ├── Adaptive weights + trust slider
    │   ├── Claude mission planner (async)
    │   ├── Mistral drone reasoner (async)
    │   └── Chat handler (NL → commands)
    └── WebSocket server (state broadcast + command ingestion)
```

### Module boundaries (strict)

- `backend/` — pure Python. Owns all simulation state. Never imports frontend.
- `frontend/` — pure TypeScript / Three.js. Never runs sim logic. Renders received state.
- [`shared/protocol.md`](shared/protocol.md) — the WebSocket schema. Source of truth for both sides.

---

## Code Tour — 10-Minute Read

If you want to understand the system top-down:

1. [`backend/src/agents/priority_market.py`](backend/src/agents/priority_market.py) — 200 lines. The unifying abstraction. Read `bid()` and `clear_market()`.
2. [`backend/src/agents/coordinator.py`](backend/src/agents/coordinator.py) — look at `_run_priority_market` and `_invalidate_targets_in_wind_gusts` for the is_in_avoid composition.
3. [`backend/src/simulation/drone.py`](backend/src/simulation/drone.py) — `update_drone_physics` and `update_drone_battery`. Notice how `wind_fn` and `height_fn` follow the same optional-callable pattern.
4. [`backend/src/agents/adaptive_weights.py`](backend/src/agents/adaptive_weights.py) — how outcomes feed back into source scales.
5. [`backend/src/agents/chat_handler.py`](backend/src/agents/chat_handler.py) — Claude translates natural language into the same `PriorityAsset` stream the operator uses.
6. [`shared/protocol.md`](shared/protocol.md) — the full wire schema.

---

## Running & Testing

```bash
make run           # Backend + frontend
make backend       # Backend only (WebSocket server at :8765)
make frontend      # Frontend only (Vite dev server at :5173)
make headless      # Run the sim with no browser
make test          # 304 backend tests
make typecheck     # ruff + tsc
make format        # ruff + prettier
```

Frontend tests:

```bash
cd frontend && npx vitest run   # 11 tests
```

Headless wind-drift repro:

```bash
cd backend && uv run python -m scripts.wind_repro
```

---

## Performance

| Metric | Value |
|---|---|
| Sim tick rate (headless) | 1,400+ ticks/sec (~70× real-time) |
| WebSocket update | ~4 KB per tick |
| Chunked world size | 10,240 m × 10,240 m |
| Drone fleet render | single draw call (`InstancedMesh`) |
| Frontend frame budget | < 8 ms at 60 FPS |
| Backend tests | 304 pass, ~6 minutes full suite |

---

## Controls

| Action | Input |
|---|---|
| Orbit camera | Mouse drag |
| Zoom | Scroll wheel |
| Pan | WASD |
| Select drone | Left-click |
| Send selected drone | Right-click terrain |
| Paint priority zone | `Z`, then drag (toggle `high` / `low` / `avoid`) |
| Chat | `T` (Escape to close) |
| Pause / resume | Space |
| Speed 1× / 2× / 5× | `1` / `2` / `3` |
| New simulation | `N` |
| Hold position | `X` |
| Return to base | `B` |

---

## Design Philosophy

- **Immutable state between ticks.** Every tick produces a new `WorldState`. Determinism for free; debuggability for free; replay for free.
- **Seeded randomness.** Terrain, survivor placement, wind, gust regions all take a seed. A run is fully reproducible.
- **Optional-callable plumbing.** Chunked-world support, wind, and wind hazards all thread through sim functions as optional `Callable`s — no coupling between layers.
- **One substrate, many producers.** Operator input, LLM strategy, environmental hazards, and outcome feedback all flow through one priority market. Adding a modality is a one-line composition.
- **Graceful degradation.** No LLM key? System runs on the classical tier. API errors? Fall back to cached decisions. The simulation never breaks.

---

## Research Areas Touched

- Multi-agent coordination under partial observability
- Bayesian search theory (PoC updates on failed scans + evidence geometry)
- Auction-based task allocation
- Human-AI teaming with adaptive weight learning
- Communication-constrained distributed agents
- Hybrid reactive / deliberative agent architectures
- Real-time simulation + browser visualization at scale

---

Built with [Claude Code](https://claude.ai/claude-code). The project is also a reference implementation of Claude Code's tooling ecosystem (skills, hooks, hierarchical `CLAUDE.md`, sub-agents).
