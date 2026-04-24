"""Core simulation engine — the fixed-timestep tick loop.

Each call to `tick()` takes an immutable WorldState and returns a new one.
No mutation. All side effects (events, discoveries) are captured in the returned state.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from src.simulation.drone import (
    WindFn,
    check_drone_failures,
    compute_comms_links,
    create_drone_fleet,
    detect_evidence,
    detect_survivors,
    update_drone_battery,
    update_drone_physics,
)
from src.simulation.survivors import update_survivors
from src.simulation.types import (
    FOG_EXPLORED,
    FOG_UNEXPLORED,
    Command,
    Drone,
    DroneStatus,
    EventType,
    Evidence,
    SimConfig,
    SimEvent,
    Vec3,
    WorldState,
)
from src.terrain.generator import generate_terrain


def create_world(config: SimConfig) -> WorldState:
    """Initialize a new simulation world from config."""
    terrain = generate_terrain(config)
    base = Vec3(terrain.width / 2.0, 0.0, terrain.height / 2.0)
    drones = create_drone_fleet(config.drone_count, base, config)
    fog_grid = np.full((terrain.height, terrain.width), FOG_UNEXPLORED, dtype=np.int8)

    return WorldState(
        tick=0,
        elapsed=0.0,
        terrain=terrain,
        drones=drones,
        survivors=terrain.survivors,
        fog_grid=fog_grid,
        comms_links=(),
        events=(),
        base_position=base,
        tick_rate=config.tick_rate,
    )


def tick(
    world: WorldState,
    dt: float,
    commands: list[Command] | None = None,
    rng: np.random.Generator | None = None,
    config: SimConfig | None = None,
    wind_fn: WindFn | None = None,
) -> WorldState:
    """Advance the simulation by one timestep.

    Args:
        world: Current immutable world state.
        dt: Time delta in seconds (typically 1/tick_rate).
        commands: Optional list of commands from humans/agents.
        rng: Random generator for failure checks. If None, no random failures.
        config: Simulation config. Required for battery/failure updates.

    Returns:
        New WorldState with all updates applied.
    """
    if config is None:
        config = SimConfig()
    if commands is None:
        commands = []

    events: list[SimEvent] = []
    new_tick = world.tick + 1
    new_elapsed = world.elapsed + dt

    # --- 1. Apply commands to drones ---
    drones = list(world.drones)
    for cmd in commands:
        drones = _apply_command(drones, cmd)

    # --- 2. Update drone physics ---
    heightmap = world.terrain.heightmap
    drones = [update_drone_physics(d, dt, heightmap, config, wind_fn=wind_fn) for d in drones]

    # --- 3. Update drone batteries ---
    updated_drones: list[Drone] = []
    for d in drones:
        old_status = d.status
        d = update_drone_battery(d, dt, config, wind_fn=wind_fn)

        if d.status == DroneStatus.FAILED and old_status != DroneStatus.FAILED:
            events.append(SimEvent(EventType.DRONE_FAILED, new_tick, drone_id=d.id))
        elif d.status == DroneStatus.RETURNING and old_status == DroneStatus.ACTIVE:
            events.append(SimEvent(EventType.DRONE_BATTERY_CRITICAL, new_tick, drone_id=d.id))
            # Set target to base
            d = replace(d, target=world.base_position)

        updated_drones.append(d)
    drones = updated_drones

    # --- 4. Random failure checks ---
    if rng is not None:
        new_drones: list[Drone] = []
        for d in drones:
            old_comms = d.comms_active
            d = check_drone_failures(d, rng, config)
            if not d.comms_active and old_comms:
                events.append(SimEvent(EventType.DRONE_COMMS_LOST, new_tick, drone_id=d.id))
            new_drones.append(d)
        drones = new_drones

    # --- 4b. Update mobile survivors (wander + flee from nearby drones) ---
    if rng is not None:
        updated_survivors = update_survivors(
            world.survivors, world.terrain, dt, rng, drones=tuple(drones),
        )
    else:
        updated_survivors = world.survivors

    # --- 5. Detect survivors ---
    survivors = list(updated_survivors)
    for d in drones:
        if d.status == DroneStatus.FAILED:
            continue
        detected_ids = detect_survivors(d, tuple(survivors), world.terrain.biome_map, config=config)
        for sid in detected_ids:
            survivors[sid] = replace(
                survivors[sid],
                discovered=True,
                discovered_by=d.id,
                discovered_at_tick=new_tick,
            )
            events.append(
                SimEvent(
                    EventType.SURVIVOR_FOUND,
                    new_tick,
                    drone_id=d.id,
                    survivor_id=sid,
                )
            )

    # --- 6. Compute communication links ---
    drones_tuple = tuple(drones)
    comms_links = compute_comms_links(drones_tuple)

    # --- 7. Update fog-of-war ---
    fog_grid = _update_fog(
        world.fog_grid, drones_tuple, config, new_tick, world.terrain.width, world.terrain.height
    )

    # --- 8. Handle returning drones that reach base ---
    final_drones: list[Drone] = []
    for d in drones:
        if d.status == DroneStatus.RETURNING and d.target is not None:
            dist_to_base = (d.position - world.base_position).length_xz()
            if dist_to_base < 5.0:  # within 5m of base
                d = replace(
                    d,
                    status=DroneStatus.RECHARGING,
                    velocity=Vec3(0, 0, 0),
                    target=None,
                    current_task="recharging",
                )
                events.append(SimEvent(EventType.DRONE_RETURNED, new_tick, drone_id=d.id))
        elif d.status == DroneStatus.RECHARGING:
            # Recharge at 2% per second
            new_battery = min(100.0, d.battery + 2.0 * dt)
            if new_battery >= 99.0:
                d = replace(
                    d,
                    battery=100.0,
                    status=DroneStatus.ACTIVE,
                    current_task="idle",
                )
            else:
                d = replace(d, battery=new_battery)
        final_drones.append(d)

    return WorldState(
        tick=new_tick,
        elapsed=new_elapsed,
        terrain=world.terrain,
        drones=tuple(final_drones),
        survivors=tuple(survivors),
        fog_grid=fog_grid,
        comms_links=comms_links,
        events=tuple(events),
        base_position=world.base_position,
        tick_rate=world.tick_rate,
    )


def _apply_command(drones: list[Drone], cmd: Command) -> list[Drone]:
    """Apply a single command to the drone list."""
    if cmd.type == "move_to" and cmd.drone_id is not None and cmd.target is not None:
        idx = cmd.drone_id
        if 0 <= idx < len(drones) and drones[idx].status == DroneStatus.ACTIVE:
            drones[idx] = replace(
                drones[idx],
                target=cmd.target,
                current_task="moving",
            )

    elif cmd.type == "return_to_base" and cmd.drone_id is not None:
        idx = cmd.drone_id
        if 0 <= idx < len(drones) and drones[idx].status == DroneStatus.ACTIVE:
            # Without target=cmd.target, physics keeps flying to the stale
            # waypoint (e.g., a mid-sweep ring point) and the drone hovers
            # there until battery depletes.
            new_target = cmd.target if cmd.target is not None else drones[idx].target
            drones[idx] = replace(
                drones[idx],
                status=DroneStatus.RETURNING,
                current_task="returning",
                target=new_target,
            )

    elif cmd.type == "search_area" and cmd.drone_id is not None and cmd.target is not None:
        idx = cmd.drone_id
        if 0 <= idx < len(drones) and drones[idx].status == DroneStatus.ACTIVE:
            drones[idx] = replace(
                drones[idx],
                target=cmd.target,
                current_task="searching",
            )

    elif cmd.type == "hold_position" and cmd.drone_id is not None:
        idx = cmd.drone_id
        if 0 <= idx < len(drones) and drones[idx].status == DroneStatus.ACTIVE:
            # Target = current position → physics hovers in place.
            # current_task="holding" is the sticky signal coordinator watches.
            drones[idx] = replace(
                drones[idx],
                target=drones[idx].position,
                current_task="holding",
            )

    return drones


def _update_fog(
    fog_grid: np.ndarray,
    drones: tuple[Drone, ...],
    config: SimConfig,
    current_tick: int,
    terrain_width: int,
    terrain_height: int,
) -> np.ndarray:
    """Update the fog-of-war grid based on drone sensor coverage.

    Returns a NEW array (no mutation of input).
    """
    new_fog = fog_grid.copy()

    # Age existing explored cells toward stale
    # We track this simply: explored cells stay explored as long as a drone is nearby.
    # For stale tracking, we'd need a "last seen" grid. For MVP, we mark cells
    # as explored when a drone can see them, and leave them explored.
    # Stale logic will be added when we have the "last seen tick" overlay.

    for drone in drones:
        if drone.status == DroneStatus.FAILED or not drone.sensor_active:
            continue

        # Drone's position in grid coordinates
        cx = int(drone.position.x)
        cz = int(drone.position.z)
        radius = int(math.ceil(drone.sensor_range))

        # Bounding box for the sensor circle
        r_min = max(0, cz - radius)
        r_max = min(terrain_height, cz + radius + 1)
        c_min = max(0, cx - radius)
        c_max = min(terrain_width, cx + radius + 1)

        # Create coordinate grids for the bounding box
        rows = np.arange(r_min, r_max)
        cols = np.arange(c_min, c_max)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        # Distance from drone (XZ plane)
        dist_sq = (cc - cx) ** 2 + (rr - cz) ** 2
        within_range = dist_sq <= drone.sensor_range**2

        new_fog[r_min:r_max, c_min:c_max] = np.where(
            within_range, FOG_EXPLORED, new_fog[r_min:r_max, c_min:c_max]
        )

    return new_fog


def get_coverage_pct(fog_grid: np.ndarray) -> float:
    """Return the percentage of terrain that has been explored (not unexplored)."""
    total = fog_grid.size
    explored = np.count_nonzero(fog_grid != FOG_UNEXPLORED)
    return (explored / total) * 100.0


# ---------------------------------------------------------------------------
# Chunked world support
# ---------------------------------------------------------------------------


def tick_chunked(
    world: WorldState,
    dt: float,
    chunked_world: object,  # ChunkedWorld
    commands: list[Command] | None = None,
    rng: np.random.Generator | None = None,
    config: SimConfig | None = None,
    wind_fn: WindFn | None = None,
) -> WorldState:
    """Advance simulation by one timestep using chunked terrain lookups.

    Same logic as ``tick()`` but uses the ChunkedWorld for height and biome
    lookups instead of a monolithic numpy array.
    """
    from src.terrain.chunked import ChunkedWorld

    cw: ChunkedWorld = chunked_world  # type: ignore[assignment]

    if config is None:
        config = SimConfig()
    if commands is None:
        commands = []

    events: list[SimEvent] = []
    new_tick = world.tick + 1
    new_elapsed = world.elapsed + dt

    world_size = cw.get_world_size()
    bounds = (world_size, world_size)

    # --- 1. Apply commands to drones ---
    drones = list(world.drones)
    for cmd in commands:
        drones = _apply_command(drones, cmd)

    # --- 2. Update drone physics (chunked height lookup) ---
    drones = [
        update_drone_physics(
            d, dt, None, config,
            height_fn=cw.get_heightmap_at,
            world_bounds=bounds,
            wind_fn=wind_fn,
        )
        for d in drones
    ]

    # --- 3. Update drone batteries ---
    updated_drones: list[Drone] = []
    for d in drones:
        old_status = d.status
        d = update_drone_battery(d, dt, config, wind_fn=wind_fn)

        if d.status == DroneStatus.FAILED and old_status != DroneStatus.FAILED:
            events.append(SimEvent(EventType.DRONE_FAILED, new_tick, drone_id=d.id))
        elif d.status == DroneStatus.RETURNING and old_status == DroneStatus.ACTIVE:
            events.append(SimEvent(EventType.DRONE_BATTERY_CRITICAL, new_tick, drone_id=d.id))
            d = replace(d, target=world.base_position)

        updated_drones.append(d)
    drones = updated_drones

    # --- 4. Random failure checks ---
    if rng is not None:
        new_drones: list[Drone] = []
        for d in drones:
            old_comms = d.comms_active
            d = check_drone_failures(d, rng, config)
            if not d.comms_active and old_comms:
                events.append(SimEvent(EventType.DRONE_COMMS_LOST, new_tick, drone_id=d.id))
            new_drones.append(d)
        drones = new_drones

    # --- 4b. Gather survivors from active chunks ---
    drone_positions = [d.position for d in drones if d.status != DroneStatus.FAILED]
    active_chunks = cw.get_active_chunks(drone_positions)
    all_survivors: list = []
    for chunk in active_chunks:
        all_survivors.extend(chunk.survivors)


    # Update mobile survivors (wander + flee from nearby drones)
    if rng is not None:
        survivors_tuple = tuple(all_survivors)
        # Use lightweight terrain stub for survivor updates
        survivors_tuple = update_survivors(
            survivors_tuple, world.terrain, dt, rng, drones=tuple(drones),
        )
    else:
        survivors_tuple = tuple(all_survivors)

    # --- 5. Detect survivors (chunked biome lookup) ---
    survivors = list(survivors_tuple)
    survivor_by_id = {s.id: i for i, s in enumerate(survivors)}
    for d in drones:
        if d.status == DroneStatus.FAILED:
            continue
        detected_ids = detect_survivors(
            d, tuple(survivors),
            biome_fn=cw.get_biome_at,
            height_fn=cw.get_heightmap_at,
            config=config,
        )
        for sid in detected_ids:
            idx = survivor_by_id.get(sid)
            if idx is not None:
                survivors[idx] = replace(
                    survivors[idx],
                    discovered=True,
                    discovered_by=d.id,
                    discovered_at_tick=new_tick,
                )
                events.append(
                    SimEvent(EventType.SURVIVOR_FOUND, new_tick, drone_id=d.id, survivor_id=sid)
                )
                # Update chunk survivor state
                for chunk in active_chunks:
                    for ci, cs in enumerate(chunk.survivors):
                        if cs.id == sid:
                            chunk.survivors[ci] = survivors[idx]

    # --- 5b. Detect evidence (clues planted by the mission) ---
    # Each newly-discovered clue triggers a Bayesian posterior update on
    # the PoC grid below (step 7b). Keep the evidence list immutable-style
    # by rebuilding it.
    evidence_list: list[Evidence] = list(world.evidence)
    evidence_found_this_tick: list[Evidence] = []
    if evidence_list:
        evidence_by_id = {e.id: i for i, e in enumerate(evidence_list)}
        for d in drones:
            if d.status == DroneStatus.FAILED:
                continue
            detected_ids = detect_evidence(
                d,
                tuple(evidence_list),
                biome_fn=cw.get_biome_at,
                config=config,
            )
            for eid in detected_ids:
                idx = evidence_by_id.get(eid)
                if idx is None:
                    continue
                e_old = evidence_list[idx]
                if e_old.discovered:
                    continue
                e_new = replace(
                    e_old,
                    discovered=True,
                    discovered_by=d.id,
                    discovered_at_tick=new_tick,
                )
                evidence_list[idx] = e_new
                evidence_found_this_tick.append(e_new)
                events.append(
                    SimEvent(
                        EventType.EVIDENCE_FOUND,
                        new_tick,
                        drone_id=d.id,
                        data={
                            "evidence_id": e_new.id,
                            "kind": e_new.kind,
                            "position": [e_new.position.x, e_new.position.y, e_new.position.z],
                            "confidence": e_new.confidence,
                            "heading": e_new.heading,
                            "age_hours": e_new.age_hours,
                        },
                    )
                )

    # --- 6. Compute communication links ---
    drones_tuple = tuple(drones)
    comms_links = compute_comms_links(drones_tuple)

    # --- 7. Update fog-of-war per-chunk + global coarse grid ---
    for chunk in active_chunks:
        _update_chunk_fog(chunk, drones_tuple, config)

    # --- 7b. Bayesian update: Probability of Containment (PoC) ---
    # For each drone that scanned without finding, decrease PoC in scanned cells.
    # PoD per drone depends on its sensor, altitude, and biome — we use a
    # simplified model here and let Phase 4 refine it.
    search_map = world.search_map
    if search_map is not None:
        from src.simulation.search_map import SearchMap
        sm: SearchMap = search_map  # type: ignore[assignment]
        for d in drones:
            if d.status == DroneStatus.FAILED or not d.sensor_active:
                continue
            # Skip if the drone just discovered a survivor this tick — that cell
            # should NOT be PoC-reduced (the target is there, not absent).
            found_this_tick = any(
                e.drone_id == d.id and e.type == EventType.SURVIVOR_FOUND
                for e in events
            )
            if found_this_tick:
                continue
            # Effective PoD: base sensor efficacy. Phase 4 will compute true
            # PoD from biome + sensor loadout + weather.
            base_pod = 0.4
            sm.update_after_failed_scan(
                center_world=(d.position.x, d.position.z),
                radius_meters=d.sensor_range,
                pod=base_pod,
            )

        # Evidence posterior: each discovery re-shapes the PoC toward the
        # geometry implied by the clue (cone for footprints, ring for
        # debris, tight circle for signal fires).
        for e in evidence_found_this_tick:
            sm.update_on_evidence(
                position=(e.position.x, e.position.z),
                kind=e.kind,
                confidence=e.confidence,
                heading=e.heading,
            )

    # Update the coarse global fog grid (used by agent AI)
    fog_grid = world.fog_grid
    fog_h, fog_w = fog_grid.shape
    fog_scale = cw.get_world_size() / max(fog_w, 1)
    new_fog = fog_grid.copy()
    for drone in drones:
        if drone.status == DroneStatus.FAILED or not drone.sensor_active:
            continue
        fog_cx = int(drone.position.x / fog_scale)
        fog_cz = int(drone.position.z / fog_scale)
        fog_r = max(1, int(math.ceil(drone.sensor_range / fog_scale)))
        r_min = max(0, fog_cz - fog_r)
        r_max = min(fog_h, fog_cz + fog_r + 1)
        c_min = max(0, fog_cx - fog_r)
        c_max = min(fog_w, fog_cx + fog_r + 1)
        if r_min < r_max and c_min < c_max:
            new_fog[r_min:r_max, c_min:c_max] = FOG_EXPLORED

    # Compute global coverage from active chunks
    total_cells = 0
    explored_cells = 0
    for chunk in active_chunks:
        total_cells += chunk.fog_grid.size
        explored_cells += np.count_nonzero(chunk.fog_grid != FOG_UNEXPLORED)

    # --- 8. Handle returning drones that reach base ---
    final_drones: list[Drone] = []
    for d in drones:
        if d.status == DroneStatus.RETURNING and d.target is not None:
            dist_to_base = (d.position - world.base_position).length_xz()
            if dist_to_base < 5.0:
                d = replace(
                    d,
                    status=DroneStatus.RECHARGING,
                    velocity=Vec3(0, 0, 0),
                    target=None,
                    current_task="recharging",
                )
                events.append(SimEvent(EventType.DRONE_RETURNED, new_tick, drone_id=d.id))
        elif d.status == DroneStatus.RECHARGING:
            new_battery = min(100.0, d.battery + 2.0 * dt)
            if new_battery >= 99.0:
                d = replace(d, battery=100.0, status=DroneStatus.ACTIVE, current_task="idle")
            else:
                d = replace(d, battery=new_battery)
        final_drones.append(d)

    return WorldState(
        tick=new_tick,
        elapsed=new_elapsed,
        terrain=world.terrain,
        drones=tuple(final_drones),
        survivors=tuple(survivors),
        fog_grid=new_fog,
        comms_links=comms_links,
        events=tuple(events),
        base_position=world.base_position,
        tick_rate=world.tick_rate,
        search_map=world.search_map,
        evidence=tuple(evidence_list),
    )


def _update_chunk_fog(
    chunk: object,  # TerrainChunk
    drones: tuple[Drone, ...],
    config: SimConfig,
) -> None:
    """Update fog-of-war for a single chunk in-place."""
    from src.terrain.chunked import TerrainChunk

    tc: TerrainChunk = chunk  # type: ignore[assignment]
    cs = tc.fog_grid.shape[0]  # chunk size
    origin_x = tc.coord.cx * cs
    origin_z = tc.coord.cz * cs

    for drone in drones:
        if drone.status == DroneStatus.FAILED or not drone.sensor_active:
            continue

        # Drone position in chunk-local coordinates
        local_x = drone.position.x - origin_x
        local_z = drone.position.z - origin_z
        radius = int(math.ceil(drone.sensor_range))

        # Bounding box (clipped to chunk)
        r_min = max(0, int(local_z) - radius)
        r_max = min(cs, int(local_z) + radius + 1)
        c_min = max(0, int(local_x) - radius)
        c_max = min(cs, int(local_x) + radius + 1)

        if r_min >= r_max or c_min >= c_max:
            continue

        rows = np.arange(r_min, r_max)
        cols = np.arange(c_min, c_max)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        dist_sq = (cc - local_x) ** 2 + (rr - local_z) ** 2
        within_range = dist_sq <= drone.sensor_range ** 2

        tc.fog_grid[r_min:r_max, c_min:c_max] = np.where(
            within_range, FOG_EXPLORED, tc.fog_grid[r_min:r_max, c_min:c_max]
        )
