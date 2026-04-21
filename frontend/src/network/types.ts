/** Types matching the WebSocket protocol in shared/protocol.md */

export interface DroneState {
  id: number;
  position: [number, number, number]; // [x, y, z]
  velocity: [number, number, number];
  heading: number;
  battery: number;
  status: "active" | "returning" | "recharging" | "failed";
  sensor_active: boolean;
  comms_active: boolean;
  current_task: string;
  target: [number, number, number] | null;
}

export interface SurvivorState {
  id: number;
  position: [number, number, number];
  discovered: boolean;
  discovered_by: number | null;
  discovered_at_tick: number | null;
}

export interface TerrainData {
  width: number;
  height: number;
  max_elevation: number;
  encoding?: string;
  /** Base64-encoded uint16 heightmap (row-major, 0-65535 → 0-max_elevation) */
  heightmap_b64?: string;
  /** Base64-encoded uint8 biome map (row-major, values 0-5) */
  biome_map_b64?: string;
  /** Legacy JSON arrays (for backwards compat) */
  heightmap?: number[][];
  biome_map?: number[][];
}

/** Decoded terrain with 2D arrays ready for rendering. */
export interface DecodedTerrain {
  width: number;
  height: number;
  max_elevation: number;
  heightmap: Float32Array;
  biome_map: Uint8Array;
}

/** Decode terrain from either base64 binary or legacy JSON format. */
export function decodeTerrain(data: TerrainData): DecodedTerrain {
  const { width, height, max_elevation } = data;

  let heightmap: Float32Array;
  let biome_map: Uint8Array;

  if (data.encoding === "base64_uint16_uint8" && data.heightmap_b64 && data.biome_map_b64) {
    // Decode base64 binary
    const hmBytes = Uint8Array.from(atob(data.heightmap_b64), (c) => c.charCodeAt(0));
    const hmUint16 = new Uint16Array(hmBytes.buffer);
    heightmap = new Float32Array(hmUint16.length);
    for (let i = 0; i < hmUint16.length; i++) {
      heightmap[i] = (hmUint16[i] / 65535) * max_elevation;
    }

    const bmBytes = Uint8Array.from(atob(data.biome_map_b64), (c) => c.charCodeAt(0));
    biome_map = new Uint8Array(bmBytes.buffer);
  } else if (data.heightmap && data.biome_map) {
    // Legacy JSON arrays
    heightmap = new Float32Array(width * height);
    biome_map = new Uint8Array(width * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        heightmap[row * width + col] = data.heightmap[row][col];
        biome_map[row * width + col] = data.biome_map[row][col];
      }
    }
  } else {
    // Empty fallback
    heightmap = new Float32Array(width * height);
    biome_map = new Uint8Array(width * height);
  }

  return { width, height, max_elevation, heightmap, biome_map };
}

/** Probability-of-Containment grid for Bayesian search overlay. */
export interface PoCGrid {
  size: number;          // Grid is size × size
  world_size: number;    // Total world size in meters
  peak: number;          // Peak PoC value (pre-quantization)
  data_b64: string;      // uint8, each cell = byte 0-255 normalized to peak
}

export interface FogData {
  width: number;
  height: number;
  rle: [number, number][]; // [value, count] pairs
}

export interface SimEvent {
  type: string;
  tick: number;
  drone_id: number | null;
  survivor_id: number | null;
}

/** A clue discovered by a drone. Drives the posterior update + marker render. */
export interface EvidenceMarker {
  id: number;
  kind: "footprint" | "debris" | "signal_fire" | string;
  position: [number, number, number];
  confidence: number;
  heading: number | null;
  age_hours: number | null;
  discovered_by: number | null;
  discovered_at_tick: number | null;
}

/** One-shot message fired the tick a drone finds evidence. */
export interface EvidenceDiscovered {
  type: "evidence_discovered";
  tick: number;
  id: number;
  kind: "footprint" | "debris" | "signal_fire" | string;
  position: [number, number, number];
  confidence: number;
  heading: number | null;
  age_hours: number | null;
  drone_id: number | null;
}

/** A user-drawn priority zone (rectangular, XZ polygon). */
export interface ZoneData {
  zone_id: string;
  polygon: [number, number][];
  priority: "high" | "low" | "avoid";
  created_tick: number;
}

export interface StateUpdate {
  type: "state_update";
  tick: number;
  elapsed: number;
  terrain?: TerrainData;
  drones: DroneState[];
  survivors: SurvivorState[];
  /** All survivors (including undiscovered) — for God Mode debug overlay. */
  all_survivors?: SurvivorState[];
  /** Probability-of-Containment grid (downsampled, sent every N ticks). */
  poc_grid?: PoCGrid;
  /** Discovered evidence on the map. Empty for missions without an evidence trail. */
  evidence?: EvidenceMarker[];
  fog_grid: FogData;
  comms_links: [number, number][];
  events: SimEvent[];
  coverage_pct: number;
  agent_info?: AgentInfo;
  /** Operator-drawn priority zones (biases coordinator's target scoring). */
  zones?: ZoneData[];
}

export interface WeatherInfo {
  wind_speed: number;
  wind_direction: number;
  gusting: boolean;
}

export interface DayCycleInfo {
  time_of_day: number;
  sun_intensity: number;
  sun_color: [number, number, number];
  sensor_effectiveness: number;
  phase: string;
}

/** "Latest" snapshot — per-tick scalars. */
export interface MetricsLatest {
  elapsed: number;
  coverage_pct: number;
  survivors_found: number;
  total_survivors: number;
  active_drones: number;
  avg_battery: number;
  entropy: number | null;
}

/** Tight "is this a good run?" signals. */
export interface MetricsSearchQuality {
  mttd_seconds: number | null;
  survival_window_pct: number | null;
  entropy_drop_pct: number | null;
}

export interface MetricsEvidenceProgress {
  discovered: number;
  planted: number;
}

export interface MetricsResources {
  total_drone_km: number;
  active_fraction: number | null;
}

export interface MetricsInfo {
  latest: MetricsLatest;
  time_to_first_discovery: number | null;
  time_to_full_coverage: number | null;
  time_to_first_evidence: number | null;
  evidence_to_survivor_latency: number | null;
  event_counts: Record<string, number>;
  efficiency_score: number | null;
  coverage_rate: number | null;
  search_quality: MetricsSearchQuality;
  evidence_progress: MetricsEvidenceProgress;
  resources: MetricsResources;
}

export interface HazardInfo {
  id: number;
  type: "no_fly_zone" | "signal_jammer";
  center: [number, number, number];
  radius: number;
}

export interface ActivityLogEntry {
  tick: number;
  elapsed: number;
  drone_id: number | null;
  message: string;
  category: "info" | "decision" | "alert" | "event";
}

export interface AgentInfo {
  phase: string;
  briefing: string;
  planner_calls: number;
  reasoner_calls: number;
  weather?: WeatherInfo;
  daycycle?: DayCycleInfo;
  metrics?: MetricsInfo;
  hazards?: HazardInfo[];
  activity_log?: ActivityLogEntry[];
}

export interface IntelPin {
  label: string;
  kind: string;
  position: [number, number];
  radius?: number;
}

export interface MissionScenario {
  name: string;
  title: string;
  description: string;
  known_facts: string[];
  base_position: [number, number, number];
  survival_window_seconds: number;
  intel_pins: IntelPin[];
}

/** Scenario briefing — sent on connect and after every reset. */
export interface MissionBriefingScenario {
  type: "mission_briefing";
  mission: MissionScenario;
  available: string[];
}

/** Strategic directive — sent when the Claude planner issues a new directive. */
export interface MissionBriefingDirective {
  type: "mission_briefing";
  tick: number;
  briefing: string;
  zone_priorities: Record<string, string>;
  reasoning: string;
}

export type MissionBriefing = MissionBriefingScenario | MissionBriefingDirective;

export function isScenarioBriefing(
  msg: MissionBriefing,
): msg is MissionBriefingScenario {
  return (msg as MissionBriefingScenario).mission !== undefined;
}

/** Chunk terrain data from the chunked world system. */
export interface ChunkTerrainData {
  type: "chunk_terrain";
  cx: number;
  cz: number;
  origin_x: number;
  origin_z: number;
  size: number;
  /** Resolution of heightmap/biome arrays (e.g. 256 when downsampled from 1024). Defaults to size. */
  resolution?: number;
  max_elevation: number;
  heightmap_b64: string;
  biome_map_b64: string;
  encoding: string;
  survivor_count: number;
  fog_rle?: FogData;
}

/** World overview for minimap (1 pixel per chunk). */
export interface WorldOverview {
  type: "world_overview";
  chunks_x: number;
  chunks_z: number;
  world_size: number;
  chunk_size: number;
  overview_rgb_b64: string;
  encoding: string;
}

export type ServerMessage =
  | StateUpdate
  | MissionBriefing
  | ChunkTerrainData
  | WorldOverview
  | EvidenceDiscovered;
