"""Core simulation engine — the fixed-timestep tick loop.

Each call to `tick()` takes an immutable WorldState and returns a new one.
No mutation. All side effects (events, discoveries) are captured in the returned state.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from src.simulation.drone import (
    check_drone_failures,
    compute_comms_links,
    create_drone_fleet,
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
    drones = [update_drone_physics(d, dt, heightmap, config) for d in drones]

    # --- 3. Update drone batteries ---
    updated_drones: list[Drone] = []
    for d in drones:
        old_status = d.status
        d = update_drone_battery(d, dt, config)

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

    # --- 4b. Update mobile survivors (random walk) ---
    if rng is not None:
        updated_survivors = update_survivors(world.survivors, world.terrain, dt, rng)
    else:
        updated_survivors = world.survivors

    # --- 5. Detect survivors ---
    survivors = list(updated_survivors)
    for d in drones:
        if d.status == DroneStatus.FAILED:
            continue
        detected_ids = detect_survivors(d, tuple(survivors))
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
            drones[idx] = replace(
                drones[idx],
                status=DroneStatus.RETURNING,
                current_task="returning",
            )

    elif cmd.type == "search_area" and cmd.drone_id is not None and cmd.target is not None:
        idx = cmd.drone_id
        if 0 <= idx < len(drones) and drones[idx].status == DroneStatus.ACTIVE:
            drones[idx] = replace(
                drones[idx],
                target=cmd.target,
                current_task="searching",
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
