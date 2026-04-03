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

export interface AgentInfo {
  phase: string;
  briefing: string;
  planner_calls: number;
  reasoner_calls: number;
}

export interface MissionBriefing {
  type: "mission_briefing";
  tick: number;
  briefing: string;
  zone_priorities: Record<string, string>;
  reasoning: string;
}

export type ServerMessage = StateUpdate | MissionBriefing;
