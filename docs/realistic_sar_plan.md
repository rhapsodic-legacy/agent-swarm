# Realistic SAR Simulation — Implementation Plan

> **Goal**: Transform the current proof-of-concept into a realistic search-and-rescue simulation that models real-world drone challenges, makes the drone's perspective the primary view, and uses visual storytelling so a viewer immediately understands what's happening without explanation.

> **Design Principle**: *Show the map as the drones see it.* Everything the viewer sees should reflect what the swarm knows, doesn't know, and is deciding. The human reviewer is a fly on the wall of an autonomous operation.

---

## Visual Layer Philosophy

Every simulation feature must have a visual counterpart. If it's not visible, it doesn't exist for the demo. The map is a living tactical display:

| Sim State | Visual Representation |
|-----------|----------------------|
| Unexplored area | Dense fog-of-war (dark overlay) |
| Explored, no findings | Clear terrain, fog lifted |
| Ruled out (impossible terrain) | Red-tinted overlay with ✗ pattern |
| High-priority zone | Pulsing amber border |
| Active search sector | Faint grid lines showing the assigned pattern |
| Detection candidate (low confidence) | Small yellow ? marker |
| Detection candidate (high confidence) | Larger orange ! marker |
| Confirmed survivor | Bright green ✓ beacon |
| Drone-to-drone communication | Brief text bubble floating between drones |
| Drone-to-MCP communication | Text bubble with uplink icon |
| Wind field | Subtle animated arrows on terrain surface |
| No-fly weather zone | Swirling red/purple overlay |
| Communication dead zone | Static/noise texture overlay |
| Drone low battery warning | Drone mesh blinks yellow |
| Drone lost/crashed | Red X at last known position |

**Toggle: "God Mode"** — Human reviewer can press a key to reveal all survivor positions (bright markers through fog). Press again to hide. This is clearly labeled as a debug/review overlay, not the drone's view.

**Toggle: "Comms View"** — Shows mesh network topology, relay chains, dead zones. Text bubbles appear showing drone communications in human-readable form.

---

## Implementation Phases

### Phase A: Detection Confidence & Unknown Survivor Count
**Priority: Highest — changes the entire search dynamic**

#### A1: Replace Binary Detection with Confidence Model

**Backend: `src/simulation/detection.py` (new file)**

```
DetectionEvent:
    drone_id: int
    position: Vec3
    tick: int
    confidence: float          # 0.0 - 1.0
    category: str              # "thermal_signature", "visual_match", "movement",
                               # "signal", "debris", "tracks", "false_positive"
    evidence: str              # Human-readable description
    sensor_type: str           # "visual", "thermal", "lidar"
    requires_investigation: bool
```

Confidence is computed from:
- **Base detection probability**: sensor type × distance × altitude
- **Environment modifier**: canopy_coverage × shadow_factor × weather_visibility × time_of_day_factor
- **Target modifier**: survivor_mobility × clothing_visibility × signaling_behavior
- **Result**: `confidence = base × env_modifier × target_modifier + noise`

Thresholds:
- `>0.85` → Auto-confirmed. Drone circles for multi-angle verification, marks GPS.
- `0.5 - 0.85` → Investigation requested. Second drone dispatched for cross-sensor check.
- `0.2 - 0.5` → Logged, queued for human/MCP review. Drone continues primary mission.
- `<0.2` → Logged silently. Noise.

#### A2: Unknown Survivor Count + Probability of Area (POA)

Replace the current `survivor_count` config with:

```
SearchMission:
    last_known_position: Vec3 | None       # If known (e.g. distress signal origin)
    estimated_party_size: tuple[int, int]   # Range, e.g. (3, 8)
    time_since_last_contact: float          # Hours — affects spread radius
    terrain_context: str                    # "hiking_trail", "plane_crash", "avalanche"
    search_area_radius: float               # Computed from time × max_travel_speed
```

**POA Grid** (`src/simulation/poa.py`, new file):
- Each cell has a `probability_of_area` score (0.0-1.0) representing likelihood of containing a survivor
- Initialized from: last known position (gaussian falloff), terrain (trails get higher POA, cliffs get lower), biome (shelter-capable terrain gets higher POA)
- Updated over time: survivors drift downhill and toward water, so POA shifts
- **POD** (Probability of Detection) per cell: how likely a pass would detect someone there, given sensor, altitude, conditions
- **POS** (Probability of Success) = POA × POD — this drives search prioritization
- Cells are never "done". POS decreases with each pass but never reaches zero.

**Termination**: No auto-stop. The MCP/coordinator decides when to recall based on:
- Cumulative POS across all cells (overall search effectiveness)
- Time elapsed vs survival window
- Remaining drone resources
- Diminishing returns curve

#### A3: Visual Layer for Confidence

**Frontend additions:**
- Detection markers rendered as instanced meshes at detection positions
  - Yellow `?` for low confidence (0.2-0.5)
  - Orange `!` for medium confidence (0.5-0.85)
  - Green `✓` for confirmed (>0.85)
  - Size scales with confidence
  - Pulse animation, fade after timeout unless confirmed
- POA heatmap overlay: warm colors (red/orange) for high probability cells, cool (blue) for low, transparent for near-zero. Toggle on/off.
- "Ruled out" overlay: cells classified as impossible (vertical cliff, deep water, >45° slope) get a red-tinted overlay with subtle cross-hatch pattern

---

### Phase B: Hierarchical Search with Altitude
**Priority: High — visually dramatic, shows strategic AI**

#### B1: Multi-Altitude Search Model

Drones operate at three altitude tiers:

| Tier | Altitude | Swath Width | Resolution | Use Case |
|------|----------|-------------|------------|----------|
| High | 200-400m | 400-800m | Low — rules out terrain, spots large signals | Initial sweep, area classification |
| Medium | 50-150m | 100-300m | Medium — identifies candidates | Sector search, systematic coverage |
| Low | 10-30m | 20-60m | High — confirms detections | Investigation, close inspection |

**Backend: Extend drone state**
```
Drone:
    ...existing fields...
    search_tier: "high" | "medium" | "low"
    assigned_altitude: float
    effective_swath: float      # Computed from altitude + sensor
    detection_resolution: float # Inversely proportional to altitude
```

The coordinator assigns tiers:
1. **Phase 1 — Classification sweep**: Send 3-5 drones at high altitude to classify the entire search area. Mark cells as: impossible, low-priority, medium-priority, high-priority.
2. **Phase 2 — Sector search**: Assign drone teams to high/medium priority sectors at medium altitude. Systematic patterns within each sector.
3. **Phase 3 — Investigation**: When a candidate is detected, dispatch a low-altitude drone to investigate. This drone breaks from its search pattern, descends, orbits the candidate, and reports back.

#### B2: Visual Layer for Altitude Tiers

- Drones at different tiers are visually distinct:
  - High: smaller (farther away), wider sensor cone, blue-tinted
  - Medium: standard size, standard cone, green-tinted
  - Low: larger (closer to camera), narrow cone, amber-tinted
- When a drone transitions altitude, animate the descent/ascent
- Sector assignments visible as colored grid overlay on terrain (each sector = a color, assigned drones have matching color on their HUD tag)
- Active search pattern visible as faint dashed lines ahead of the drone (shows where it's planning to go)

---

### Phase C: Drone Communications & Visual Text
**Priority: High — the demo's storytelling mechanism**

#### C1: Communication System Overhaul

**Backend: `src/simulation/comms.py` (overhaul existing)**

```
CommsMessage:
    from_id: int               # Drone ID or -1 for MCP
    to_id: int | None          # Specific drone, or None for broadcast
    channel: "drone-drone" | "drone-mcp" | "mcp-drone"
    content: str               # Human-readable version of the message
    data: dict                 # Machine-readable payload
    tick: int
    requires_relay: bool       # True if sender can't directly reach recipient
    relay_chain: list[int]     # Drone IDs that relayed this message
```

Message types that generate visible text:
- **Detection report**: "Thermal hit, confidence 0.62. Requesting visual confirmation." (drone → MCP)
- **Investigation dispatch**: "Drone 7, investigate candidate at [4521, 3892]. Descend to 25m, circle pattern." (MCP → drone)
- **Status update**: "Sector B4 swept. 0 candidates. Moving to B5." (drone → MCP)
- **Relay request**: "Signal lost to base. Drone 12, can you relay?" (drone → drone)
- **Weather warning**: "Wind exceeding safe threshold at ridge. Altitude hold at 180m." (drone → MCP)
- **Confirmation**: "Confirmed survivor at [4523, 3889]. Conscious, waving. Dispatching coordinates." (drone → MCP)
- **Resource alert**: "Battery at 22%. RTB in 90 seconds." (drone → MCP)
- **Pattern reasoning**: "Three low-confidence thermal hits in a line heading NE. Consistent with group movement. Elevating sector priority." (MCP → all)

#### C2: Visual Text System

**Frontend: `src/ui/commsOverlay.ts` (new file)**

Text bubbles appear in 3D space:
- Positioned above the sending drone
- CSS2DRenderer or sprite-based for always-facing-camera text
- Color-coded by channel:
  - White: drone-to-drone
  - Cyan: drone-to-MCP
  - Gold: MCP-to-drone (directives)
  - Red: alerts/warnings
- Fade in over 0.3s, hold for 3-5s (configurable), fade out over 1s
- If the viewer is zoomed out, text auto-scales to remain readable
- Maximum 5-8 visible bubbles at once to avoid clutter (queue + prioritize by importance)

**Comms log panel** (bottom-right, expandable):
- Scrolling feed of all messages
- Filterable by drone, by channel, by type
- Timestamps relative to mission clock
- This is the "full picture" — the 3D bubbles are the highlights

#### C3: Communication Dead Zones

- Terrain blocks line-of-sight radio. Compute which cells have LoS to base and to relay drones.
- Dead zone overlay: static/noise texture where drones lose contact
- Drones entering dead zones switch to autonomous mode. Their last known plan is visible (dashed line), but their real-time position is unknown until they re-establish contact or a relay drone moves into range.
- When a drone re-establishes contact, a burst of queued messages appears

---

### Phase D: Weather, Wind & Environmental Hazards
**Priority: Medium-high — makes terrain matter tactically**

#### D1: Terrain-Aware Wind Field

**Backend: `src/simulation/wind.py` (new or overhaul `weather.py`)**

Replace global wind vector with a terrain-influenced wind field:
- Base wind: direction + speed (changes slowly over time)
- **Ridgeline acceleration**: wind speed × 1.5-3x at ridges perpendicular to wind direction
- **Lee-side turbulence**: random gusts on the downwind side of ridges
- **Valley channeling**: wind follows valley floors, intensified
- **Thermal updrafts**: sun-facing slopes during daytime generate lift (reduced battery drain for drones riding updrafts)

Wind field is computed as a coarse grid (e.g. 64x64 for the world) and interpolated.

**Drone interaction with wind:**
- Headwind reduces ground speed, increases battery drain
- Tailwind increases ground speed, reduces drain
- Crosswind above threshold → drone drifts, must correct (increased drain)
- Gusts above safe threshold → drone must hold position or descend
- Wind field informs routing: coordinator plans paths to exploit tailwinds and avoid ridge-top turbulence

#### D2: Weather Conditions as Search Modifiers

Extend existing weather system with:
- **Precipitation type + intensity**: clear, light rain, heavy rain, snow, blizzard
- **Fog/cloud ceiling**: altitude above which (or below which) visibility drops to near-zero
- **Icing altitude**: above this altitude, rotor icing occurs. Drone must descend or land.
- **Temperature**: affects battery performance (below -10°C, capacity drops 30-50%)

Each condition modifies detection probability per the sensor effectiveness table from the brainstorm.

#### D3: Visual Layer for Weather

- **Wind arrows**: small animated arrows on the terrain surface showing wind direction and relative speed. Denser/longer arrows = stronger wind. Subtle — shouldn't overwhelm terrain.
- **No-fly zones**: areas where wind exceeds safe threshold get a swirling red-purple overlay that animates. Drones route around these.
- **Precipitation**: particle system for rain/snow (Three.js Points). Density varies by zone.
- **Fog**: volumetric fog (or FogExp2 with spatially varying density) that obscures terrain in low-lying areas.
- **Cloud ceiling**: a translucent plane at the ceiling altitude. Drones above it are dimmed/hidden.

---

### Phase E: Sensor Realism & False Positives
**Priority: Medium — adds depth to the confidence system**

#### E1: Multi-Sensor Loadout

Each drone carries 1-2 sensors (weight constraint):
```
SensorLoadout:
    visual: bool      # Standard camera. Good in daylight, useless at night.
    thermal: bool     # IR camera. Good at night, noisy in warm conditions.
    lidar: bool       # Point cloud. Works in fog/rain, no color info.
```

Fleet composition matters: coordinator assigns mixed teams so thermal drones sweep wide, visual drones investigate hits.

#### E2: False Positive Generator

**Backend: `src/simulation/false_positives.py` (new file)**

Generate plausible false positive detections:
- **Wildlife**: thermal signatures at animal-appropriate locations (forest, meadow). Tagged as "thermal_signature", confidence 0.3-0.6.
- **Terrain artifacts**: sun-heated rocks, reflective surfaces. Visual matches with confidence 0.2-0.4.
- **Old campsites**: debris near trails. Visual match, confidence 0.4-0.6.
- **Water reflection**: thermal/visual noise near water bodies.

False positive rate: ~5-15 per real survivor. This makes the confidence system and human-in-the-loop triage genuinely necessary.

#### E3: Line-of-Sight & Canopy

**Backend: `src/simulation/los.py` (new file)**

- **Canopy density**: forest biome cells have canopy_coverage (0.6-0.95). Reduces visual detection proportionally. Thermal partially penetrates (30-50% reduction instead of 60-95%).
- **Terrain shadow**: at current time-of-day, compute shadow direction from sun angle. Cells in shadow: visual detection -40-70%.
- **Canyon/ravine occlusion**: cells with steep neighbors on 3+ sides are effectively invisible from overhead passes. Require side-approach at low altitude.

#### E4: Visual Layer for Sensor Effects

- When a drone scans a cell, brief highlight showing effective detection probability:
  - Green tint: good detection conditions (>60% effective)
  - Yellow tint: degraded conditions (30-60%)
  - Red tint: poor conditions (<30%)
- Canopy rendered as darker/denser terrain texture in forest biomes
- Shadow map overlay that shifts with time of day (real-time shadow casting from sun position)

---

### Phase F: Survivor Behavior & Time Pressure
**Priority: Medium — adds narrative and urgency**

#### F1: Survivor Agent Model

**Backend: `src/simulation/survivor.py` (overhaul)**

```
Survivor:
    ...existing fields...
    mobility: "mobile" | "slow" | "immobile"
    heading: float                 # Direction of travel (if mobile)
    speed: float                   # m/s (0 for immobile, 0.5-1.5 for mobile)
    health: float                  # 0.0 - 1.0, decreases over time
    signaling: bool                # Actively trying to be found
    has_transponder: bool          # Electronic beacon
    shelter_type: str | None       # "cave", "tree_cover", "tent", "snow_cave", None
    group_id: int | None           # Survivors may be in groups
    behavior: str                  # "following_trail", "seeking_shelter", "stationary", "signaling"
```

Behavior model:
- **Mobile survivors**: walk 1-3 km/h. Tend toward trails, water, downhill. May build signal fires.
- **Injured**: very slow or stationary. Seek nearest shelter. Signaling degrades over time.
- **Groups**: may split (some go for help, others stay). Group members cluster.
- **Time effects**: health degrades (exposure, dehydration). Below 0.3 health → immobile. Below 0.1 → unconscious (no signaling, reduced thermal signature).
- **Response to drones**: mobile survivors wave, use mirrors, build fires when they hear/see a drone. This *increases* detection probability when drones are nearby.

#### F2: Survival Window & Mission Clock

- Prominent mission clock on HUD showing elapsed time
- Survival probability curve: starts at 95%, drops based on conditions
  - Himalayan winter: drops to 50% at 24h, 20% at 48h, 5% at 72h
  - Temperate forest: drops to 50% at 48h, 20% at 96h
- MCP uses survival window to modulate risk tolerance: as time runs out, accept more aggressive search patterns, lower confidence thresholds for investigation

#### F3: Visual Layer for Survivors

- Mobile survivors leave faint tracks on terrain (visible to low-altitude drones)
- Signal fires produce a small particle effect + thermal bloom
- Health visualized on confirmed survivor markers (green → yellow → red beacon)
- Groups shown as clustered markers with a count badge
- When "God Mode" toggle is on, all survivors visible with their health and behavior state

---

### Phase G: Drone Attrition & Cold Weather Effects
**Priority: Lower — adds realism depth**

#### G1: Cold Battery Model
- Battery capacity = `base_capacity × temperature_factor`
- At -10°C: 70% capacity. At -20°C: 50%. At -30°C: 35%.
- Drones at high altitude in cold conditions must RTB earlier.
- Visual: battery indicator on drone HUD tag turns blue when cold-degraded.

#### G2: Failure Modes
- **Icing**: above icing altitude, rotors accumulate ice. Performance degrades over time. Drone must descend or crash.
- **GPS denial**: deep valleys cause position drift. Drone switches to dead reckoning (inertial). Position uncertainty grows over time — shown as a growing circle around the drone's estimated position.
- **Collision**: low-altitude flight near terrain in wind has a crash probability per meter. Higher near trees, cliffs, narrow canyons.
- **Sensor failure**: individual sensors can fail (cold, impact, moisture). Drone continues with remaining sensors at reduced capability.

#### G3: Visual Layer for Attrition
- Crashed drones: red X marker at crash site with "LOST" label. Last known data dump shown if unrecovered.
- GPS uncertainty: translucent circle around drone that grows when in GPS-denied area, shrinks when back in coverage.
- Icing warning: drone mesh gets a white frost overlay, blinks when critical.

---

### Phase H: Area Scale & Forward Operating Bases
**Priority: Lower — extends the simulation scope**

#### H1: Larger Search Area
- Increase world to 50km x 50km (or configurable)
- Forward Operating Bases (FOBs): deployable recharge stations at strategic locations
- Drones RTB to nearest FOB, not just the starting base
- Relay drones: some drones assigned as stationary relay nodes on high points

#### H2: Visual Layer for Scale
- Minimap becomes essential at this scale
- FOBs shown as landing pad icons with recharge status
- Relay drones shown with radio wave animation
- Search progress bar showing overall area coverage percentage

---

## Frontend UI Overhaul Summary

### New Toggle Controls (keyboard shortcuts + buttons)

| Key | Toggle | Description |
|-----|--------|-------------|
| G | God Mode | Show all survivor positions through fog |
| C | Comms View | Show communication text bubbles + mesh network |
| H | Heatmap | Show POA probability overlay |
| W | Wind | Show wind field arrows |
| Z | Zones | Show sector assignments + ruled-out areas |
| F | Fog | Toggle fog-of-war (always on by default) |
| D | Detection | Show all detection candidates with confidence |

### HUD Additions

```
┌─────────────────────────┐
│ MISSION STATUS           │
│ Elapsed: 2:34:17         │
│ Survival Window: 71%     │ ← color-coded urgency
│ Drones: 17/20 active     │
│ Confirmed: 3             │
│ Investigating: 5         │
│ Candidates: 12           │
│ Ruled Out: 34km²         │
│ Coverage (effective): 41% │
│ Phase: Sector Search      │
│ Weather: Snow, -12°C      │
│ Wind: 28 km/h NW          │
│ Icing Alt: >4200m         │
└─────────────────────────┘
```

### Comms Log Panel
- Expandable side panel (press L or click)
- Scrolling feed of drone ↔ MCP messages
- Filter buttons: All | Detections | Directives | Alerts
- Each entry: `[02:31:15] Drone 7 → MCP: Thermal hit, conf 0.58...`

---

## Protocol Additions (shared/protocol.md)

New message types needed:

### `detection_event` (Server → Client)
```json
{
    "type": "detection_event",
    "drone_id": 7,
    "position": [4521.0, 120.0, 3892.0],
    "confidence": 0.62,
    "category": "thermal_signature",
    "evidence": "Thermal anomaly 2.1°C above ambient. Size consistent with human.",
    "sensor_type": "thermal",
    "tick": 5400
}
```

### `comms_message` (Server → Client)
```json
{
    "type": "comms_message",
    "from_id": 7,
    "to_id": -1,
    "channel": "drone-mcp",
    "content": "Thermal hit, confidence 0.62. Requesting visual confirmation.",
    "tick": 5400
}
```

### `zone_update` (Server → Client)
```json
{
    "type": "zone_update",
    "zones": [
        {
            "id": "sector_B4",
            "status": "ruled_out",
            "reason": "vertical_cliff",
            "bounds": [3072, 4096, 4096, 5120]
        },
        {
            "id": "sector_C2",
            "status": "high_priority",
            "poa_score": 0.73,
            "assigned_drones": [3, 7, 12]
        }
    ]
}
```

### `poa_update` (Server → Client)
```json
{
    "type": "poa_update",
    "resolution": 64,
    "data_b64": "...",
    "encoding": "base64_float16"
}
```

---

## Implementation Order

This is designed so each phase produces a demoable improvement:

```
Phase A  (confidence + unknown count)     → search feels real, not a game
Phase C  (comms + visual text)            → viewer understands what drones think
Phase B  (hierarchical altitude)          → dramatic visual of strategic AI
Phase E  (sensor realism + false positives) → confidence system becomes necessary  
Phase D  (weather + wind)                 → terrain becomes tactical
Phase F  (survivor behavior)              → adds narrative tension
Phase G  (drone attrition)                → adds stakes and strategy
Phase H  (area scale + FOBs)             → full operational scope
```

Each phase should take 1-3 sessions. Frontend visual layer is built alongside each backend phase, not deferred.

---

## README Note for Final Demo

> **What you're seeing**: This visualization shows the search operation *from the swarm's perspective*. Fog-of-war represents areas the drones haven't explored. Detection markers show what the drones have found and how confident they are. The text communications between drones are simplified for human readability — in a real system, these would be compact binary messages. The "God Mode" toggle (G key) reveals ground truth for comparison.
