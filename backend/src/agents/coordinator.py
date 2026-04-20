"""Decentralized multi-agent coordinator.

Each drone maintains a local belief map and makes decisions based on:
1. Its own sensor data
2. Information shared by nearby drones (via comms links)
3. Its assigned zone and search pattern

The coordinator runs every tick, producing Commands for each active drone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.agents.biome_profiles import (
    BiomeFlightProfile,
    get_profile_at_position,
)
from src.agents.drone_reasoner import DroneReasoner
from src.agents.mission_planner import MissionPlanner
from src.agents.pathfinding import (
    potential_field_direction,
)
from src.agents.search_patterns import (
    frontier_search,
    lawnmower_waypoints,
    priority_search,
)
from src.simulation.types import (
    Command,
    Drone,
    DroneStatus,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)


class SearchPhase(Enum):
    """High-level mission phase for the swarm."""

    INITIAL_SPREAD = auto()  # drones fanning out to assigned zones
    SYSTEMATIC_SEARCH = auto()  # lawnmower sweep within zones
    FRONTIER_HUNT = auto()  # frontier-based exploration of remaining gaps
    PRIORITY_SWEEP = auto()  # focus on high-value unexplored areas
    COMPLETE = auto()  # full coverage achieved


class DroneTask(Enum):
    """Per-drone task state."""

    IDLE = auto()
    MOVING_TO_ZONE = auto()
    SWEEPING = auto()
    FRONTIER_EXPLORING = auto()
    PRIORITY_SEARCHING = auto()
    INVESTIGATING = auto()  # heading to a point of interest
    RETURNING_TO_BASE = auto()


@dataclass
class DroneAgent:
    """Mutable per-drone agent state (lives outside the immutable sim state)."""

    drone_id: int
    task: DroneTask = DroneTask.IDLE
    zone: tuple[tuple[int, int], tuple[int, int]] | None = None
    waypoints: list[Vec3] | None = None
    waypoint_index: int = 0
    current_target: Vec3 | None = None
    ticks_at_target: int = 0  # how long we've been near current target
    # Local belief: which survivors this drone knows about (shared via comms)
    known_survivors: set[int] | None = None
    # Hotspot sweep queue: when the drone arrives at a PoC hotspot, we queue
    # perimeter waypoints so it sweeps the surrounding cluster before
    # ping-ponging to the next hotspot across the map.
    local_sweep_waypoints: list[Vec3] | None = None


def _generate_local_sweep_waypoints(
    center: Vec3,
    inner_radius: float = 80.0,
    outer_radius: float = 180.0,
) -> list[Vec3]:
    """Two hexagonal rings around a PoC hotspot.

    Survivor clusters are 50-300m across and survivors drift well beyond
    cluster centers (observed up to 1km drift in simulation). Sensor range
    is 40m, so a drone landing on a hotspot cell scans one 40m disc and
    leaves — it misses everyone in the cluster.

    Two rings (inner 80m / outer 180m) with 6 points each cover a donut
    from ~40m to ~220m around the hotspot. Outer ring is rotated 30° so
    scan footprints interlock. 12 waypoints × ~100m each = ~1200m sweep,
    manageable on a single battery cycle at ~10 m/s cruise.
    """
    import math

    waypoints: list[Vec3] = []
    # Inner ring: 6 points at 0°, 60°, 120°, ...
    for k in range(6):
        angle = k * math.pi / 3.0
        x = center.x + math.cos(angle) * inner_radius
        z = center.z + math.sin(angle) * inner_radius
        waypoints.append(Vec3(x, center.y, z))
    # Outer ring: 6 points at 30°, 90°, 150°, ... (offset for interlocking)
    for k in range(6):
        angle = (k * math.pi / 3.0) + (math.pi / 6.0)
        x = center.x + math.cos(angle) * outer_radius
        z = center.z + math.sin(angle) * outer_radius
        waypoints.append(Vec3(x, center.y, z))
    return waypoints


class SwarmCoordinator:
    """Runs classical AI coordination for the drone fleet.

    Call `update(world, config)` each tick to get a list of Commands.
    Maintains mutable agent state separate from the immutable WorldState.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.agents: dict[int, DroneAgent] = {}
        self.phase = SearchPhase.INITIAL_SPREAD
        self.zones_assigned = False
        self.tick_count = 0
        # Phase transition thresholds (coverage %)
        self.systematic_threshold = 5.0  # switch from spread to systematic
        self.frontier_threshold = 70.0  # switch to frontier hunting
        self.priority_threshold = 90.0  # switch to priority sweep
        # LLM agents
        self.mission_planner = MissionPlanner()
        self.drone_reasoner = DroneReasoner()
        # Activity log — ring buffer of recent entries for frontend display
        self._activity_log: list[dict] = []
        self._max_log_entries = 200
        # Per-tick cache for PoC-based target selection
        self._poc_cache_tick: int = -1
        self._poc_hottest: list[tuple[int, int, float]] = []
        self._poc_claimed: set[tuple[int, int]] = set()

    def _log(self, tick: int, elapsed: float, drone_id: int | None, message: str, category: str = "info") -> None:
        """Append an entry to the activity log ring buffer."""
        entry = {
            "tick": tick,
            "elapsed": round(elapsed, 1),
            "drone_id": drone_id,
            "message": message,
            "category": category,  # "info", "decision", "alert", "event"
        }
        self._activity_log.append(entry)
        if len(self._activity_log) > self._max_log_entries:
            self._activity_log = self._activity_log[-self._max_log_entries:]

    def get_recent_log(self, since_tick: int = 0) -> list[dict]:
        """Return log entries since the given tick (for incremental frontend updates)."""
        return [e for e in self._activity_log if e["tick"] >= since_tick]

    def update(self, world: WorldState, config: SimConfig) -> list[Command]:
        """Produce commands for all active drones based on current world state."""
        self.tick_count += 1
        commands: list[Command] = []

        # Ensure all drones have agent state
        for drone in world.drones:
            if drone.id not in self.agents:
                self.agents[drone.id] = DroneAgent(drone_id=drone.id, known_survivors=set())

        # Assign zones on first call
        if not self.zones_assigned:
            self._assign_zones(world)
            self.zones_assigned = True
            active = sum(1 for d in world.drones if d.status == DroneStatus.ACTIVE)
            self._log(world.tick, world.elapsed, None, f"Zones assigned to {active} drones. Beginning spread.", "decision")

        # Update mission phase based on coverage
        coverage = self._get_coverage(world.fog_grid)
        old_phase = self.phase
        self._update_phase(coverage)
        if self.phase != old_phase:
            self._log(world.tick, world.elapsed, None,
                      f"Phase: {old_phase.name} → {self.phase.name} (coverage {coverage:.1f}%)", "decision")

        # Share knowledge between connected drones
        self._share_knowledge(world)

        # --- Log significant events ---
        evidence_this_tick = 0
        for event in world.events:
            if event.type.name == "SURVIVOR_FOUND":
                self._log(world.tick, world.elapsed, event.drone_id,
                          f"Survivor #{event.survivor_id} found!", "event")
            elif event.type.name == "DRONE_FAILED":
                self._log(world.tick, world.elapsed, event.drone_id,
                          "Drone failed — battery depleted.", "alert")
            elif event.type.name == "DRONE_RETURNED":
                self._log(world.tick, world.elapsed, event.drone_id,
                          "Returned to base. Recharging.", "info")
            elif event.type.name == "DRONE_COMMS_LOST":
                self._log(world.tick, world.elapsed, event.drone_id,
                          "Comms lost — out of range.", "alert")
            elif event.type.name == "EVIDENCE_FOUND":
                evidence_this_tick += 1
                data = event.data or {}
                kind = data.get("kind", "evidence")
                pos = data.get("position", [0, 0, 0])
                self._log(
                    world.tick, world.elapsed, event.drone_id,
                    f"Evidence: {kind} at ({pos[0]:.0f}, {pos[2]:.0f}). Updating search.",
                    "event",
                )

        # When new evidence comes in, the PoC posterior has already been
        # re-shaped inside the engine (search_map.update_on_evidence).
        # Bust the hotspot cache so the next target selection samples the
        # new posterior instead of the stale pre-evidence one.
        if evidence_this_tick > 0:
            self._poc_cache_tick = -1
            self._poc_hottest = []
            self._poc_claimed = set()

        # --- LLM Layer: Mission Planner (Claude) ---
        for event in world.events:
            self.mission_planner.record_event(
                event.type.name,
                f"drone={event.drone_id} survivor={event.survivor_id}",
            )

        if self.mission_planner.should_plan(world):
            self.mission_planner.trigger_plan(world)
            self._log(world.tick, world.elapsed, None, "Mission planner re-evaluating strategy...", "decision")

        directive = self.mission_planner.consume_result()
        if directive:
            self._apply_mission_directive(directive, world)
            briefing = directive.get("briefing", "")
            if briefing:
                self._log(world.tick, world.elapsed, None, f"MCP: {briefing}", "decision")

        # --- LLM Layer: Drone Reasoner (Mistral) ---
        for drone in world.drones:
            if self.drone_reasoner.should_reason(drone, world.tick):
                self.drone_reasoner.trigger_reasoning(drone, world)

        llm_decisions = self.drone_reasoner.consume_decisions()

        # Generate commands for each active drone
        for drone in world.drones:
            if drone.status in (DroneStatus.FAILED, DroneStatus.RECHARGING):
                continue

            if drone.status == DroneStatus.RETURNING:
                continue

            # Human-issued hold — respect until the operator issues a new command.
            if drone.current_task == "holding":
                continue

            agent = self.agents[drone.id]

            # Smart battery check
            if drone.status == DroneStatus.ACTIVE and self._should_return_for_fuel(drone, world):
                agent.task = DroneTask.RETURNING_TO_BASE
                agent.current_target = world.base_position
                agent.local_sweep_waypoints = None
                commands.append(
                    Command(type="return_to_base", drone_id=drone.id, target=world.base_position)
                )
                self._log(world.tick, world.elapsed, drone.id,
                          f"RTB — battery {drone.battery:.0f}%, need fuel for return trip.", "alert")
                continue

            # LLM decision
            llm_decision = llm_decisions.get(drone.id)
            if llm_decision:
                cmd = self._apply_llm_decision(drone, agent, llm_decision, world)
                if cmd is not None:
                    commands.append(cmd)
                    reasoning = llm_decision.get("reasoning", llm_decision.get("action", ""))
                    self._log(world.tick, world.elapsed, drone.id,
                              f"AI decision: {reasoning}", "decision")
                    continue

            # Classical AI
            cmd = self._decide(drone, agent, world)
            if cmd is not None:
                commands.append(cmd)

        self._handle_failures(world)

        return commands

    def _apply_mission_directive(self, directive: dict, world: WorldState) -> None:
        """Apply strategic directives from the mission planner."""
        # Recall low-battery drones
        for drone_id in directive.get("recall_drones", []):
            if drone_id in self.agents:
                agent = self.agents[drone_id]
                agent.task = DroneTask.RETURNING_TO_BASE
                agent.current_target = world.base_position

        # Apply zone priority shifts (the coordinator will naturally
        # pick higher-priority zones in its search logic)

    def _apply_llm_decision(
        self, drone: Drone, agent: DroneAgent, decision: dict, world: WorldState
    ) -> Command | None:
        """Convert an LLM drone decision into a Command."""
        action = decision.get("action", "")
        target = decision.get("target")

        if action == "return_to_base":
            agent.task = DroneTask.RETURNING_TO_BASE
            return Command(type="return_to_base", drone_id=drone.id, target=world.base_position)

        if target and isinstance(target, list) and len(target) >= 2:
            tx, tz = float(target[0]), float(target[1])
            # Clamp to terrain bounds
            tw = world.terrain.width - 1
            th = world.terrain.height - 1
            tx = max(0.0, min(float(tw), tx))
            tz = max(0.0, min(float(th), tz))
            row = int(min(tz, th))
            col = int(min(tx, tw))
            elev = float(world.terrain.heightmap[row][col])
            world_target = Vec3(tx, elev, tz)

            if action == "investigate":
                agent.task = DroneTask.INVESTIGATING
            elif action == "reposition":
                agent.task = DroneTask.FRONTIER_EXPLORING
            else:
                agent.task = DroneTask.SWEEPING

            agent.current_target = world_target
            adjusted = self._apply_repulsion(drone, world_target, world)
            return Command(type="move_to", drone_id=drone.id, target=adjusted)

        return None

    def _assign_zones(self, world: WorldState) -> None:
        """Assign each drone a sector sweep radiating outward from base.

        Uses expanding sector (pie-slice) assignment instead of rectangular
        grid zones. Each drone gets an angular sector and sweeps outward from
        the base in concentric arcs — like a real SAR expanding square/sector
        search from a command post.
        """
        import math

        active_drones = [d for d in world.drones if d.status == DroneStatus.ACTIVE]
        if not active_drones:
            return

        tw = world.terrain.width
        th = world.terrain.height
        base_x = world.base_position.x
        base_z = world.base_position.z
        n = len(active_drones)

        # Each drone gets an equal angular sector
        sector_angle = 2.0 * math.pi / n

        # Initial sweep radius: start close, expand as coverage grows
        coverage = self._get_coverage(world.fog_grid)
        sweep_radius = 500 + coverage * 80  # 500m initial → ~8500m at full coverage

        for i, drone in enumerate(active_drones):
            agent = self.agents[drone.id]

            # Sector center angle for this drone
            angle = i * sector_angle

            # Target: point along the sector centerline at the current sweep radius
            target_x = base_x + math.cos(angle) * sweep_radius
            target_z = base_z + math.sin(angle) * sweep_radius

            # Clamp to world bounds
            target_x = max(50.0, min(float(tw - 50), target_x))
            target_z = max(50.0, min(float(th - 50), target_z))

            # Assign a zone that covers this sector (rough bounding box)
            half_arc = sweep_radius * math.sin(sector_angle / 2)
            zone_r_min = int(max(0, target_z - half_arc))
            zone_r_max = int(min(th, target_z + half_arc))
            zone_c_min = int(max(0, target_x - half_arc))
            zone_c_max = int(min(tw, target_x + half_arc))
            agent.zone = ((zone_r_min, zone_c_min), (zone_r_max, zone_c_max))

            agent.task = DroneTask.MOVING_TO_ZONE
            row = int(min(target_z, th - 1))
            col = int(min(target_x, tw - 1))
            elev = float(world.terrain.heightmap[row][col])
            agent.current_target = Vec3(target_x, elev, target_z)

    def _update_phase(self, coverage: float) -> None:
        """Transition between mission phases based on coverage."""
        if coverage >= 99.0:
            self.phase = SearchPhase.COMPLETE
        elif coverage >= self.priority_threshold:
            self.phase = SearchPhase.PRIORITY_SWEEP
        elif coverage >= self.frontier_threshold:
            self.phase = SearchPhase.FRONTIER_HUNT
        elif coverage >= self.systematic_threshold:
            self.phase = SearchPhase.SYSTEMATIC_SEARCH
        else:
            self.phase = SearchPhase.INITIAL_SPREAD

    def _share_knowledge(self, world: WorldState) -> None:
        """Share discovered survivor IDs between connected drones."""
        # Build adjacency from comms links
        neighbors: dict[int, set[int]] = {}
        for a_id, b_id in world.comms_links:
            neighbors.setdefault(a_id, set()).add(b_id)
            neighbors.setdefault(b_id, set()).add(a_id)

        # Each drone shares its known survivors with neighbors
        for drone_id, agent in self.agents.items():
            if agent.known_survivors is None:
                agent.known_survivors = set()
            for neighbor_id in neighbors.get(drone_id, set()):
                neighbor_agent = self.agents.get(neighbor_id)
                if neighbor_agent and neighbor_agent.known_survivors is not None:
                    agent.known_survivors |= neighbor_agent.known_survivors

        # Update from actual discovered survivors in world state
        for survivor in world.survivors:
            if survivor.discovered:
                for agent in self.agents.values():
                    if agent.known_survivors is not None:
                        agent.known_survivors.add(survivor.id)

    def _decide(self, drone: Drone, agent: DroneAgent, world: WorldState) -> Command | None:
        """Decide what a single drone should do this tick."""
        # Check if we've reached current target
        if agent.current_target is not None:
            dist = (drone.position - agent.current_target).length_xz()
            if dist < 5.0:
                agent.ticks_at_target += 1
                if agent.ticks_at_target > 10:
                    # Target reached — advance to next behavior
                    agent.current_target = None
                    agent.ticks_at_target = 0
            else:
                agent.ticks_at_target = 0

        # If we still have a valid target, keep going (with repulsion)
        if agent.current_target is not None and agent.ticks_at_target <= 10:
            adjusted = self._apply_repulsion(drone, agent.current_target, world)
            return Command(type="move_to", drone_id=drone.id, target=adjusted)

        # Need a new target — decide based on phase and task
        new_target = self._pick_target(drone, agent, world)
        if new_target is not None:
            agent.current_target = new_target
            agent.ticks_at_target = 0
            adjusted = self._apply_repulsion(drone, new_target, world)
            return Command(type="move_to", drone_id=drone.id, target=adjusted)

        return None

    def _get_survivor_positions(self, world: WorldState) -> list[Vec3]:
        """Collect world positions of all discovered survivors."""
        return [s.position for s in world.survivors if s.discovered]

    def _pick_target(self, drone: Drone, agent: DroneAgent, world: WorldState) -> Vec3 | None:
        """Pick the next target for a drone based on the current mission phase.

        If a Bayesian search map is available, use PoC-based targeting — drones
        go to the highest-probability unclaimed cells. This is the phase-1
        Bayesian search dispatcher.

        Otherwise fall back to phase-based legacy (fog + biome) targeting.
        """
        if self.phase == SearchPhase.COMPLETE:
            agent.task = DroneTask.RETURNING_TO_BASE
            agent.local_sweep_waypoints = None
            return world.base_position

        # Bayesian PoC-based routing (preferred when search map is available)
        if world.search_map is not None:
            # Continue sweeping the current cluster before picking a new hotspot
            if agent.local_sweep_waypoints:
                next_wp = agent.local_sweep_waypoints.pop(0)
                tw = world.terrain.width
                th = world.terrain.height
                tx = max(50.0, min(float(tw - 50), next_wp.x))
                tz = max(50.0, min(float(th - 50), next_wp.z))
                trow = int(min(tz, th - 1))
                tcol = int(min(tx, tw - 1))
                elev = float(world.terrain.heightmap[trow][tcol])
                agent.task = DroneTask.SWEEPING
                return Vec3(tx, elev, tz)

            poc_target = self._poc_target(drone, agent, world)
            if poc_target is not None:
                # Enqueue a local sweep — the drone will visit the hotspot
                # center first (this return), then these perimeter points
                # before selecting a new hotspot.
                agent.local_sweep_waypoints = _generate_local_sweep_waypoints(
                    poc_target
                )
                return poc_target
            # No reachable hot cell (likely low battery or far from hotspots).
            # Return to base rather than picking an unreachable target.
            agent.task = DroneTask.RETURNING_TO_BASE
            agent.local_sweep_waypoints = None
            return world.base_position

        fog_grid = world.fog_grid
        terrain = world.terrain
        surv_pos = self._get_survivor_positions(world)

        if self.phase == SearchPhase.INITIAL_SPREAD:
            agent.task = DroneTask.MOVING_TO_ZONE
            return self._zone_center_target(agent, terrain)

        if self.phase == SearchPhase.SYSTEMATIC_SEARCH:
            return self._systematic_target(drone, agent, world)

        if self.phase == SearchPhase.FRONTIER_HUNT:
            agent.task = DroneTask.FRONTIER_EXPLORING
            result = frontier_search(drone.position, fog_grid, terrain, surv_pos)
            if result is not None:
                return result
            return priority_search(drone.position, fog_grid, terrain, surv_pos)

        if self.phase == SearchPhase.PRIORITY_SWEEP:
            agent.task = DroneTask.PRIORITY_SEARCHING
            result = priority_search(drone.position, fog_grid, terrain, surv_pos)
            if result is not None:
                return result
            return frontier_search(drone.position, fog_grid, terrain, surv_pos)

        return None

    def _poc_target(self, drone: Drone, agent: DroneAgent, world: WorldState) -> Vec3 | None:
        """Select a target from DIVERSE hotspots (NMS-sampled).

        Instead of picking the top-N hottest cells (which all cluster around
        the dominant peak), we use non-maximum suppression to get one hotspot
        per cluster region. Then each drone picks the closest free hotspot.

        This produces real search-theory behavior: drones fan out across
        ALL high-probability regions simultaneously, not just the biggest one.
        """
        import math

        from src.simulation.search_map import SearchMap

        sm: SearchMap = world.search_map  # type: ignore[assignment]

        # Hotspot separation: for N drones in W×W world, ideal spacing is
        # W/sqrt(N). 20 drones in 10km → ~2300m. Separation should be smaller
        # so we have more than N hotspots (allowing drones choice), but big
        # enough that adjacent candidates are from different clusters.
        world_size = max(world.terrain.width, world.terrain.height)
        active_count = max(sum(1 for d in world.drones if d.status == DroneStatus.ACTIVE), 1)
        hotspot_separation = 0.4 * (world_size / math.sqrt(active_count))

        # Cache diverse hotspots per tick. n = 2*active so each drone has
        # options even after claims.
        if self._poc_cache_tick != world.tick:
            n_spots = active_count * 2 + 5
            self._poc_hottest = sm.diverse_hotspots(
                n_spots, min_separation_meters=hotspot_separation,
            )
            self._poc_claimed = set()
            self._poc_cache_tick = world.tick

        # Collect other drones' active targets — avoid picking their target.
        other_targets: list[tuple[float, float]] = []
        for d in world.drones:
            if d.id == drone.id or d.status != DroneStatus.ACTIVE:
                continue
            other_agent = self.agents.get(d.id)
            if other_agent is not None and other_agent.current_target is not None:
                other_targets.append(
                    (other_agent.current_target.x, other_agent.current_target.z)
                )

        # Safety: compute max total flight distance from current battery.
        # Measured actual drain: 0.58%/s = 4.83× base — use 5× for margin.
        speed_est = drone.max_speed * 0.85
        drain_est = self.config.drone_battery_drain_rate * 5.0  # %/s
        available_battery = max(drone.battery - 10.0, 0.0)
        max_flight_time = available_battery / max(drain_est, 0.01)
        max_range_meters = speed_est * max_flight_time
        # Round-trip: target must be at most half the range from CURRENT position
        # PLUS at most that much from the base.
        base_x, base_z = world.base_position.x, world.base_position.z

        # Pick the closest non-claimed, non-other-drone hotspot.
        best_cell = None
        best_score = -math.inf
        for col, row, poc_val in self._poc_hottest:
            if poc_val <= 0.0:
                continue
            if (col, row) in self._poc_claimed:
                continue
            tx, tz = sm.cell_to_world(col, row)

            # Range check: can the drone reach this target AND return to base?
            dist_to_target = math.sqrt(
                (drone.position.x - tx) ** 2 + (drone.position.z - tz) ** 2
            )
            dist_target_to_base = math.sqrt((tx - base_x) ** 2 + (tz - base_z) ** 2)
            if dist_to_target + dist_target_to_base > max_range_meters:
                continue  # can't safely reach + return

            # Skip hotspots any other drone is already targeting
            claimed_by_other = False
            for (ox, oz) in other_targets:
                if math.sqrt((tx - ox) ** 2 + (tz - oz) ** 2) < hotspot_separation * 0.5:
                    claimed_by_other = True
                    break
            if claimed_by_other:
                continue

            dist = max(dist_to_target, 1.0)

            # Score: prefer high PoC per unit distance
            score = poc_val / dist
            if score > best_score:
                best_score = score
                best_cell = (col, row)

        # Fallback: if all diverse hotspots are unreachable OR taken by others,
        # pick any unclaimed one — but STILL enforce range (never commit to
        # a target the drone can't survive). If no reachable target exists,
        # the drone will return nothing and the caller will RTB.
        if best_cell is None:
            for col, row, poc_val in self._poc_hottest:
                if (col, row) in self._poc_claimed or poc_val <= 0.0:
                    continue
                tx, tz = sm.cell_to_world(col, row)
                dist_to_target = math.sqrt(
                    (drone.position.x - tx) ** 2 + (drone.position.z - tz) ** 2
                )
                dist_target_to_base = math.sqrt(
                    (tx - base_x) ** 2 + (tz - base_z) ** 2
                )
                if dist_to_target + dist_target_to_base > max_range_meters:
                    continue
                dist = max(dist_to_target, 1.0)
                score = poc_val / dist
                if score > best_score:
                    best_score = score
                    best_cell = (col, row)

        if best_cell is None:
            return None

        self._poc_claimed.add(best_cell)
        tx, tz = sm.cell_to_world(*best_cell)
        # Clamp to world bounds and look up terrain height
        tw = world.terrain.width
        th = world.terrain.height
        tx = max(50.0, min(float(tw - 50), tx))
        tz = max(50.0, min(float(th - 50), tz))
        row = int(min(tz, th - 1))
        col = int(min(tx, tw - 1))
        elev = float(world.terrain.heightmap[row][col])
        agent.task = DroneTask.FRONTIER_EXPLORING  # closest existing task to "PoC-seeking"
        return Vec3(tx, elev, tz)

    def _systematic_target(self, drone: Drone, agent: DroneAgent, world: WorldState) -> Vec3 | None:
        """Generate waypoints for systematic lawnmower search within zone."""
        agent.task = DroneTask.SWEEPING

        # If we have waypoints, advance to next
        if agent.waypoints and agent.waypoint_index < len(agent.waypoints):
            target = agent.waypoints[agent.waypoint_index]
            agent.waypoint_index += 1
            return target

        # Generate new lawnmower waypoints for assigned zone
        # Use biome-aware spacing: tighter in forest/urban, wider in open terrain
        if agent.zone is not None:
            zone_min, zone_max = agent.zone
            profile = self._get_zone_profile(agent.zone, world)
            agent.waypoints = lawnmower_waypoints(
                zone_min,
                zone_max,
                spacing=int(profile.search_spacing),
                terrain=world.terrain,
            )
            agent.waypoint_index = 0
            if agent.waypoints:
                target = agent.waypoints[0]
                agent.waypoint_index = 1
                return target

        # Fallback to frontier search if no zone
        return frontier_search(drone.position, world.fog_grid, world.terrain)

    def _zone_center_target(self, agent: DroneAgent, terrain: Terrain) -> Vec3 | None:
        """Return the center of the drone's assigned zone."""
        if agent.zone is None:
            return None
        (r_min, c_min), (r_max, c_max) = agent.zone
        cr = min((r_min + r_max) // 2, terrain.height - 1)
        cc = min((c_min + c_max) // 2, terrain.width - 1)
        elev = float(terrain.heightmap[cr][cc])
        return Vec3(float(cc), elev, float(cr))

    def _get_zone_profile(
        self, zone: tuple[tuple[int, int], tuple[int, int]], world: WorldState
    ) -> BiomeFlightProfile:
        """Get the dominant biome flight profile for a zone."""
        (r_min, c_min), (r_max, c_max) = zone
        cr = min((r_min + r_max) // 2, world.terrain.height - 1)
        cc = min((c_min + c_max) // 2, world.terrain.width - 1)
        return get_profile_at_position(
            float(cc),
            float(cr),
            world.terrain.biome_map,
            world.terrain.width,
            world.terrain.height,
        )

    def _get_drone_profile(self, drone: Drone, world: WorldState) -> BiomeFlightProfile:
        """Get the biome flight profile at the drone's current position."""
        return get_profile_at_position(
            drone.position.x,
            drone.position.z,
            world.terrain.biome_map,
            world.terrain.width,
            world.terrain.height,
        )

    def _apply_repulsion(self, drone: Drone, target: Vec3, world: WorldState) -> Vec3:
        """Adjust a target by adding drone-repulsion to prevent clustering.

        Repulsion range adapts to biome — tighter in forest/urban (drones cluster),
        wider in open terrain.
        """
        other_positions = [
            d.position for d in world.drones if d.id != drone.id and d.status == DroneStatus.ACTIVE
        ]
        if not other_positions:
            return target

        profile = self._get_drone_profile(drone, world)
        repulsion = potential_field_direction(
            drone.position,
            other_positions,
            repulsion_range=profile.repulsion_range,
            repulsion_strength=6.0,
        )

        # Blend repulsion into target direction
        adjusted = Vec3(
            target.x + repulsion.x,
            target.y,
            target.z + repulsion.z,
        )

        # Clamp to terrain bounds
        tw = world.terrain.width - 1
        th = world.terrain.height - 1
        return Vec3(
            max(0.0, min(float(tw), adjusted.x)),
            adjusted.y,
            max(0.0, min(float(th), adjusted.z)),
        )

    def _handle_failures(self, world: WorldState) -> None:
        """Reassign failed drones' zones to nearby active drones."""
        failed_ids = {d.id for d in world.drones if d.status == DroneStatus.FAILED}
        if not failed_ids:
            return

        for fid in failed_ids:
            failed_agent = self.agents.get(fid)
            if failed_agent is None or failed_agent.zone is None:
                continue

            # Find the nearest active drone without a zone (or with complete zone)
            failed_zone = failed_agent.zone
            best_drone = None
            best_dist = float("inf")

            for drone in world.drones:
                if drone.status != DroneStatus.ACTIVE:
                    continue
                if drone.id == fid:
                    continue
                agent = self.agents[drone.id]
                # Prefer drones that have finished their zone sweep
                if agent.waypoints and agent.waypoint_index < len(agent.waypoints) - 2:
                    continue

                dist = (
                    drone.position
                    - Vec3(
                        float((failed_zone[0][1] + failed_zone[1][1]) / 2),
                        0,
                        float((failed_zone[0][0] + failed_zone[1][0]) / 2),
                    )
                ).length_xz()
                if dist < best_dist:
                    best_dist = dist
                    best_drone = drone

            if best_drone is not None:
                reassigned = self.agents[best_drone.id]
                reassigned.zone = failed_zone
                reassigned.waypoints = None
                reassigned.waypoint_index = 0
                reassigned.task = DroneTask.MOVING_TO_ZONE

            # Clear the failed drone's zone
            failed_agent.zone = None
            failed_agent.task = DroneTask.IDLE

    def _should_return_for_fuel(self, drone: Drone, world: WorldState) -> bool:
        """Proactively return a drone if it won't have enough battery to get back.

        Actual drain rate at typical search flight is base × speed_mult × alt_mult
        × sensor_mult ≈ 0.12 × 1.75 × 1.25 × 1.2 = 0.315%/s, which is 2.6x the
        base rate. The estimate here uses 3x base as a conservative safety factor.
        """
        dist_to_base = (drone.position - world.base_position).length_xz()
        if dist_to_base < 10.0:
            return False  # already near base

        # MEASURED actual drain during typical flight is ~0.58%/s for
        # base_drain=0.12 — i.e. 4.83× the base rate (altitude + acceleration
        # + sensor + velocity multipliers compound). Use 5× as safety.
        speed = max(drone.max_speed * 0.75, 1.0)  # effective ground speed
        time_to_base = dist_to_base / speed
        drain_rate = self.config.drone_battery_drain_rate * 5.0
        battery_needed = drain_rate * time_to_base

        # 10% reserve for queuing at base + minor detours
        return drone.battery < (battery_needed + 10.0)

    def _get_coverage(self, fog_grid: np.ndarray) -> float:
        """Calculate exploration coverage percentage."""
        total = fog_grid.size
        explored = np.count_nonzero(fog_grid != 0)
        return (explored / total) * 100.0
