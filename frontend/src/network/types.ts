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
  heightmap: number[][];
  biome_map: number[][];
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

export interface StateUpdate {
  type: "state_update";
  tick: number;
  elapsed: number;
  terrain?: TerrainData;
  drones: DroneState[];
  survivors: SurvivorState[];
  fog_grid: FogData;
  comms_links: [number, number][];
  events: SimEvent[];
  coverage_pct: number;
  agent_info?: AgentInfo;
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

export interface MetricsInfo {
  elapsed: number;
  coverage_pct: number;
  survivors_found: number;
  active_drones: number;
  avg_battery: number;
  efficiency_score: number;
  coverage_rate: number;
  time_to_first_discovery: number | null;
  time_to_full_coverage: number | null;
}

export interface AgentInfo {
  phase: string;
  briefing: string;
  planner_calls: number;
  reasoner_calls: number;
  weather?: WeatherInfo;
  daycycle?: DayCycleInfo;
  metrics?: MetricsInfo;
}

export interface MissionBriefing {
  type: "mission_briefing";
  tick: number;
  briefing: string;
  zone_priorities: Record<string, string>;
  reasoning: string;
}

export type ServerMessage = StateUpdate | MissionBriefing;
