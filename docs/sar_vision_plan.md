# SAR Vision — Master Plan

> **Vision**: A drone swarm that scales human search-and-rescue indefinitely. Thousands of drones with computer vision and AI coordinate to find survivors in search areas where human ground teams fail due to scale, terrain, or time pressure. The system ingests prior intelligence (crash coordinates, last-known positions, flight paths, known trails), updates its beliefs as evidence accumulates (footprints, debris, signal fires, survivor interviews), and dynamically reallocates resources to maximize the probability of finding survivors within their survival window.
>
> **Maximal scenario**: Plane crash at sea. 500 mi² search area. 1000 drones. Wind and current data updates continuously. Each new piece of evidence (AIS ping, floating debris, radar echo from satellite) narrows the search. Probability of saving any given survivor goes from <10% (current human SAR for that area) to >70%.
>
> **Minimum scenario**: Lost hiker. 50 km² of forested mountain. 20 drones. Trailhead known, last ping from phone 6 hours ago. System computes max-reach circle from that point, weights toward trails and downhill drainage, sweeps systematically at multiple altitudes.

This document supersedes neither [`general_plan.md`](../general_plan.md) (the project's original vision) nor [`realistic_sar_plan.md`](./realistic_sar_plan.md) (the realism-focused phases). It unifies them with what's already built and what the vision demands.

---

## Status: What's Already Built

| Feature | Status | Notes |
|---------|--------|-------|
| 10km x 10km chunked world | ✅ | 1024m chunks, OpenSimplex biomes, lazy generation |
| Procedural terrain (6 biomes) | ✅ | Water, beach, forest, urban, mountain, snow |
| Drone physics + battery | ✅ | Accel, cruise altitude, biome-adaptive flight profiles |
| Detection (biome + altitude + LOS + weather) | ✅ | Tunable canopy/urban occlusion, transponder beacons |
| Sector search (expanding arcs from base) | ✅ | Replaces rectangular grid zones |
| Crash-site survivor placement | ✅ | Impact cluster + debris field + walkaways + unrelated hikers |
| Fog-of-war (per-chunk + coarse global) | ✅ | Tracked, rendered, toggleable via god mode |
| God mode survivor markers | ✅ | Proper scale, unmissable red cylinders |
| Minimap | ✅ | World overview + drone positions + survivor dots |
| Weather (wind, gust) | ✅ | Global, affects flight but not search routing yet |
| Day/night cycle | ✅ | Affects lighting; not wired to sensors yet |
| LLM mission planner (Claude) | ✅ | Tier-1 coordinator; called on events |
| LLM drone reasoner (Mistral) | ✅ | Tier-2 per-drone tactics; async, rate-limited |
| Human chat interface | ✅ | Natural language overrides |
| E2E test pipeline | ✅ | Pipeline tests + live WebSocket test + Puppeteer visual agents |

**The core engine exists. The next phases are about making the search *real* — turning it from a coverage game into a search-and-rescue mission.**

---

## The Core Shift: Bayesian Search Theory

Real SAR planning is not about uniform coverage — it's about **maximizing expected probability of success per unit resource**. Every pass through a cell updates the belief about whether a survivor is there. This is the mathematical heart of the US Coast Guard's CASP system and land SAR's SARPlan — developed since the 1970s and proven in thousands of rescues.

The simulator needs three grids, each the size of the world:

| Grid | What it represents | How it's updated |
|------|--------------------|------------------|
| **PoC** (Probability of Containment) | Where the survivor likely is, given everything we know | Prior from mission type + intel; updated after each scan (Bayesian); updated on evidence finds |
| **PoD** (Probability of Detection) | If the survivor is here and a drone scans this cell now, what's the chance of detecting them? | Computed from sensor + altitude + biome + weather + time-of-day |
| **PoS** (Probability of Success) | `PoC × PoD` — what a scan actually buys us | Recomputed each tick |

The coordinator's job changes from "assign zones" to "route drones to maximize integrated PoS over remaining resources (time, battery, drone-count)". This is the actual optimization problem real SAR planners solve.

---

## Phase Plan

Phases are ordered by **demo impact** and **mathematical dependency**. Each produces a working, demoable state. The original [`realistic_sar_plan.md`](./realistic_sar_plan.md) phases are referenced by letter where they fit.

### Phase 1 — Bayesian Search Foundation (PoC / PoD / PoS grids)
**Impact**: Everything that follows rests on this. Without PoC grids, there's no "search", only coverage.
**Builds on**: Existing fog grid, detection model. Adapts [realistic_sar_plan Phase A](./realistic_sar_plan.md#phase-a-detection-confidence--unknown-survivor-count).

Steps:
1. Create `backend/src/simulation/search_map.py` with `PoCGrid`, `PoDGrid`, `PoSGrid` classes backed by NumPy arrays at configurable resolution (default 256×256 for 10km world = ~40m per cell).
2. Prior seeding interface: `set_prior_from_mission(mission: SearchMission)` — the mission describes where survivors are *likely* to be, not where they are.
3. Bayesian update on scan: `update_after_scan(cells: list[Cell], pod: float)`. The formula: `PoC_new = PoC_old × (1 - PoD) / (1 - PoC_old × PoD)` for each cell (failure-to-detect update).
4. Posterior update on evidence: `update_on_evidence(evidence)` — finding a clue increases surrounding PoC, decreases elsewhere (normalized).
5. Replace `_assign_zones()` in coordinator with `_assign_by_expected_pos()`: each drone routes to the cells with highest `PoC × PoD_if_i_scan_this`, respecting battery budget.
6. Frontend: add heatmap overlay (toggle H key) showing PoC. Bright red = high belief, blue = low, transparent = near-zero.
7. E2E test: seed a prior with known hotspots, run the sim, verify drones concentrate there and PoC decreases correctly.

Exit criteria: Watching the sim, it's obvious the drones are searching *where they should*, not just sweeping.

---

### Phase 2 — Mission Types & Prior Intelligence
**Impact**: Different scenarios look and behave fundamentally different. This is what makes it feel like a real tool, not a toy.
**New work**: Not in existing plans. Required for the "includes prior intel" vision goal.

Steps:
1. Create `backend/src/simulation/mission.py` with `SearchMission` dataclass and scenario templates:
   - `LostHikerMission`: trailhead position, time-since-last-contact, trail network, max walking speed → prior is an expanding ellipse along trails, weighted by downhill drainage.
   - `AircraftCrashMission`: last radar position, heading, altitude, airspeed, debris-dispersion cone angle → prior is a cone along flight path with Gaussian spread.
   - `MaritimeSARMission`: last known position, current vectors, wind vector, time elapsed → prior is a drift-integrated ellipse (leeway model from Coast Guard).
   - `AvalancheMission`: fracture line, runout zone, burial depth distribution → prior is concentrated on the runout fan.
   - `DisasterResponseMission`: affected-area polygon, structure density map, population density → prior weighted by urban/building density.
2. Each mission has its own `seed_poc_grid(grid)` method implementing the domain-specific prior.
3. Mission selector in settings panel (drop-down): "Plane Crash" / "Lost Hiker" / "Maritime" / "Avalanche" / "Disaster Response".
4. Each scenario sets different base configurations: world biomes (maritime = all water), survival window, survivor count distribution, drone loadout defaults.
5. Frontend: when a mission loads, show an intel briefing overlay with a 2-3 sentence scenario description and the known facts (last-known position pinned on minimap, search area boundary drawn).
6. E2E test: load each scenario, verify the prior looks right (hiker = elongated along trail; crash = cone; maritime = drift ellipse).

Exit criteria: Each scenario is visually distinct in the opening moments. A viewer can tell what kind of SAR mission this is within 5 seconds.

---

### Phase 3 — Evidence System & Posterior Updating
**Impact**: Watching the belief map re-focus in real time as evidence comes in is the most compelling part of real SAR. This is the narrative.
**New work**: Not in existing plans.

Steps:
1. Create `backend/src/simulation/evidence.py` with `Evidence` dataclass:
   ```python
   Evidence:
       tick: int
       position: Vec3
       kind: Literal["footprint", "debris", "signal_fire", "clothing",
                     "gear", "body", "beacon_ping", "vocal_signal",
                     "witness_report"]
       confidence: float
       heading: float | None   # For directional clues
       age_hours: float | None # "Footprints ~3h old" — affects update
   ```
2. Evidence-to-PoC update rules per kind:
   - `footprint` with heading → high-POA cone in travel direction, low-POA behind.
   - `debris` → high-POA ring at expected drift distance from this point given elapsed time.
   - `signal_fire` → very high-POA within 500m radius (active survivor signaling).
   - `body` → confirms an area was occupied; other group members likely nearby (cluster prior).
   - `beacon_ping` (EPIRB/PLB) → very high-POA within GPS accuracy (~50m).
   - `witness_report` (from found survivor) → domain-specific update based on their testimony.
3. Ground-truth evidence generator: during terrain generation, the mission plants clues along the "true path" survivors took from the start point to their current position. Drones discover these during flight.
4. Coordinator hooks: when evidence is discovered, it triggers an immediate posterior update and a recomputation of drone assignments.
5. Frontend: evidence markers (icons for each kind) appear on the map when discovered. A short text bubble describes what was found. The heatmap visibly re-focuses.
6. E2E test: plant a footprint at a known location with heading NE, run the sim, verify PoC increases NE of the footprint and drone assignments shift there.

Exit criteria: A viewer watching the demo can see the swarm's behavior change the moment evidence is found. The "aha" moment is visible.

---

### Phase 4 — Sensor Realism & Confidence
**Impact**: Makes detection a probability, not binary. Enables false positives and the cross-verification dynamic. (Mostly planned in [realistic_sar_plan Phase A + E](./realistic_sar_plan.md#phase-a-detection-confidence--unknown-survivor-count) — this phase ships it.)

Steps:
1. `backend/src/simulation/detection.py`: implement `DetectionEvent` with confidence, category, sensor_type.
2. Sensor loadouts: each drone has a combination (visual + thermal, visual + lidar, thermal only, etc.). Coordinator assigns mixed teams.
3. Confidence formula from realistic_sar_plan Phase A1.
4. False positive generator (`false_positives.py`): wildlife thermal signatures, sun-heated rocks, water reflections. 5-15 false positives per true positive.
5. Cross-verification: low-confidence detections request a second drone to scan from a different angle. Confidence updates based on whether the second drone corroborates.
6. Frontend: detection markers per realistic_sar_plan Phase A3 (yellow `?`, orange `!`, green `✓`).
7. E2E test: verify confidence distribution looks right and cross-verification resolves ambiguity.

Exit criteria: Most detections aren't immediately actionable — they require follow-up. This matches reality and justifies the coordinator's role.

---

### Phase 5 — Drone Communications with Visual Text
**Impact**: Viewers can *read* what the drones are thinking. Transforms the demo from silent UI to narrative.
**Adapts**: [realistic_sar_plan Phase C](./realistic_sar_plan.md#phase-c-drone-communications--visual-text).

Steps:
1. Ship realistic_sar_plan Phase C1 + C2 as written: `comms.py` with structured messages, CSS2DRenderer text bubbles above drones.
2. Auto-generate messages from simulation events (no LLM needed for most):
   - New detection → "Thermal hit at [x, y], conf 0.6, investigating."
   - Sector complete → "Sector B4 clear, moving to C2."
   - Evidence found → "Footprints heading NE, age 3h. Updating search."
   - Battery low → "RTB in 90s, battery 22%."
3. LLM (Mistral, existing) generates *tactical* messages on a lower frequency for texture: "Three thermal hits in a line heading NE. Consistent with a group. Prioritizing sector C3."
4. Comms log panel (press L).
5. Relay chains + dead zones from Phase C3.
6. E2E test: verify bubbles render without overlap issues and log panel captures everything.

Exit criteria: Can follow the mission narrative purely from reading drone communications for 30 seconds.

---

### Phase 6 — Environment as Tactical Layer
**Impact**: Terrain and weather stop being decoration and start driving decisions.
**Adapts**: [realistic_sar_plan Phase D](./realistic_sar_plan.md#phase-d-weather-wind--environmental-hazards) — ship it.

Steps per realistic_sar_plan D1-D3. Additions for maritime:
1. **Ocean currents**: for maritime scenarios, currents are the dominant drift force. Render as flow field, update survivor drift, update PoC posterior.
2. **Sea state**: affects visual detection of small objects (person in water). Choppy water = ~30% detection.
3. **Visibility**: fog banks, cloud cover, glare from sun on water.

Exit criteria: Weather routing is visible in drone paths. They fly around storm cells, ride tailwinds, descend below cloud layer.

---

### Phase 7 — Survivor Behavior & Survival Window
**Impact**: Adds time pressure. A viewer watching the clock tick feels the stakes.
**Ships**: [realistic_sar_plan Phase F](./realistic_sar_plan.md#phase-f-survivor-behavior--time-pressure).

Steps per realistic_sar_plan F1-F3. Key addition:
- Survival probability curve (hypothermia, dehydration, injury) displayed prominently. Drops over time based on weather + terrain.
- Coordinator's risk tolerance increases as the curve falls: early in the mission, only investigate high-confidence detections; late, investigate everything.

Exit criteria: The mission feels urgent. Watching the survival curve fall as drones search creates tension.

---

### Phase 8 — Scale: Forward Operating Bases + 1000-Drone Maritime
**Impact**: This is the vision shot. Showing 1000 drones searching 500 mi² is what makes the portfolio irresistible.
**Adapts**: [realistic_sar_plan Phase H](./realistic_sar_plan.md#phase-h-area-scale--forward-operating-bases).

Steps:
1. Make world size configurable. Add `MARITIME_500` preset: ~35km × 35km (~1225 km² ≈ 470 mi²).
2. Forward Operating Bases: mobile ships or shore stations where drones recharge. Multiple FOBs, drones RTB to the nearest.
3. Relay drones: loitering platforms at altitude providing comms coverage.
4. Render optimization: at 1000-drone scale, we can't draw every drone as an individual mesh. Use aggressive InstancedMesh with LOD.
5. Chunk streaming at scale: 35km × 35km with 1km chunks = 1225 chunks. Generate on demand, evict aggressively.
6. Backend scaling: may need to move from synchronous per-drone loops to vectorized NumPy operations for physics and detection.
7. Demo preset: scripted scenario where an AIS ping comes in mid-mission, narrows the search, drones redeploy.

Exit criteria: 1000 drones. 500 mi². 30+ FPS. Recognizably faster and more effective than a human search.

---

### Phase 9 — Human-in-the-Loop (Optional Operator)
**Impact**: Demonstrates human-AI teaming. The operator can do nothing (full autonomy) or guide the search.
**Adapts**: [realistic_sar_plan HUD additions and `zone_command`](./realistic_sar_plan.md#frontend-ui-overhaul-summary), [general_plan Phase 5](../general_plan.md#phase-5-human-in-the-loop-interface).

Steps:
1. Operator can draw priority polygons on the map (high/low priority zones).
2. Operator can mark last-known positions (seeds the prior mid-mission, useful when new intel arrives).
3. Operator can confirm/reject detection candidates (overrides drone confidence).
4. Operator can recall individual drones or swarms.
5. Operator can tweak mission parameters live (sensor mode, risk tolerance).
6. All operator actions are logged and feed back into the mission planner's context.
7. "Autonomy slider" (0 to 100%): at 0, every decision is proposed to the operator; at 100, fully autonomous. Proposed actions appear as ghost overlays the operator approves.

Exit criteria: Operator can productively guide the search without ever touching code. Hand someone the mouse, they can run it.

---

### Phase 10 — Polish & Portfolio
**Ships**: [general_plan Phase 7](../general_plan.md#phase-7-polish--portfolio-presentation).

Steps from general_plan Phase 7. Additions specific to this vision:
1. 4 short demo videos (30-60s each):
   - Maritime 1000-drone with incoming AIS ping
   - Lost hiker with evidence-driven belief update
   - Plane crash with debris field discovery
   - Operator-guided avalanche response
2. Side-by-side "traditional SAR vs drone swarm" comparison with time-to-find metrics.
3. A README section on the Bayesian search theory and how the PoC grid drives everything.

---

## Cross-Cutting Concerns

### Testing Discipline (lessons from the last few weeks)
Every phase adds to both:
- **Pipeline tests** (`backend/tests/test_e2e_pipeline.py`): verify state flows through the engine correctly.
- **Visual agents** (`scripts/visual_agent_verify.mjs`): independent Puppeteer-based agents that read pixels from a live browser and verify what's on screen. **Never trust code reasoning when pixels can be read.**

The `/investigate` skill documents the debugging protocol: trace actual data, don't guess.

### Performance Budgets
| Scale | Drones | Area | Tick rate | Chunks loaded | Render FPS |
|-------|--------|------|-----------|---------------|------------|
| Demo  | 20     | 10km | 10Hz      | ~10           | 30 |
| Mid   | 100    | 30km | 10Hz      | ~50           | 30 |
| Max   | 1000   | 35km | 5Hz       | ~100 active   | 30 |

At max scale, per-tick budget is 200ms. We'll need vectorized NumPy physics and detection. The coordinator's PoS optimization can run on a slower cadence (every N ticks) with drones using cached assignments between recomputations.

### Protocol Evolution
New WebSocket message types needed (additive to existing):
- `poa_update` — PoC heatmap snapshots (compressed; sent on change)
- `detection_event` — per realistic_sar_plan Phase A
- `comms_message` — per realistic_sar_plan Phase C
- `evidence_found` — new, for the evidence system
- `mission_briefing` — new, for scenario context
- `intel_update` — new, for mid-mission operator input

Keep everything additive; the frontend can ignore unknown types.

### Data-Driven Scenarios
Mission scenarios live in `backend/scenarios/*.yaml` files:
```yaml
name: maritime_aircraft_500mi
world:
  size: 35000
  chunk_size: 1024
  biome_override: all_water
  currents:
    base_vector: [0.3, 0, 0.5]   # 0.3 m/s E, 0.5 m/s N
mission:
  type: maritime_aircraft
  last_known_position: [17500, 0, 17500]
  time_since_contact: 2.5  # hours
  debris_dispersion: 0.15
  survivor_count_range: [3, 8]
  intel:
    - { t: 0, kind: last_radar_ping, pos: [17500, 17500] }
    - { t: 1800, kind: ais_beacon, pos: [18200, 17800], note: "EPIRB activation" }
drones:
  count: 1000
  loadouts:
    visual_thermal: 600
    radar: 200
    radio_beacon: 200
```
This makes new scenarios cheap to author without code changes.

---

## What This Delivers

When all phases are complete, this simulator demonstrates:

1. **Real SAR methodology**: Bayesian search theory, PoC/PoD/PoS, applied exactly as professionals do.
2. **Prior intelligence fusion**: plane crash kinematics, hiker movement models, maritime drift, avalanche runout — each scenario uses its real-world physics.
3. **Evidence-driven adaptation**: belief maps update visibly when drones find clues. Watching the swarm "realize" something new is the money shot.
4. **Sensor realism**: false positives are dominant; cross-verification is required; operators (or the mission planner LLM) triage candidates.
5. **Tactical environment**: weather, terrain, and currents drive routing decisions — not just aesthetics.
6. **Scaling demonstration**: 1000 drones on 500 mi² maritime search with the search effectiveness curve plotted against elapsed time.
7. **Human-AI teaming**: operator can do nothing (full autonomy) or guide every decision (full approval). Shows the full spectrum.
8. **Narrative clarity**: drone communications render as readable text; a viewer understands what's happening without explanation.

This is what makes the portfolio piece defensible to someone who actually works in SAR: it isn't a pretty toy. It's a faithful model of a problem with real life-and-death stakes, and it shows how autonomous swarms could change the answer.
