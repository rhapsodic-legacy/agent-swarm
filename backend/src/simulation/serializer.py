"""Serialize simulation state to JSON-compatible dicts for WebSocket broadcast.

Follows the protocol defined in shared/protocol.md.
"""

from __future__ import annotations

import numpy as np

from src.simulation.engine import get_coverage_pct
from src.simulation.types import (
    Drone,
    SimEvent,
    Survivor,
    Terrain,
    Vec3,
    WorldState,
)


def _round_vec3(v: Vec3) -> list[float]:
    return [round(v.x, 1), round(v.y, 1), round(v.z, 1)]


def serialize_state(
    world: WorldState,
    include_terrain: bool = False,
    agent_info: dict | None = None,
) -> dict:
    """Serialize a WorldState to a JSON-compatible dict.

    Args:
        world: The current simulation state.
        include_terrain: If True, include full terrain data (first message only).
        agent_info: Optional agent layer info (phase, briefing, LLM decisions).
    """
    msg: dict = {
        "type": "state_update",
        "tick": world.tick,
        "elapsed": round(world.elapsed, 2),
        "drones": [_serialize_drone(d) for d in world.drones],
        "survivors": [_serialize_survivor(s) for s in world.survivors if s.discovered],
        "fog_grid": _compress_fog_grid(world.fog_grid),
        "comms_links": [list(link) for link in world.comms_links],
        "events": [_serialize_event(e) for e in world.events],
        "coverage_pct": round(get_coverage_pct(world.fog_grid), 1),
    }

    if include_terrain:
        msg["terrain"] = _serialize_terrain(world.terrain)

    if agent_info:
        msg["agent_info"] = agent_info

    return msg


def _serialize_drone(drone: Drone) -> dict:
    return {
        "id": drone.id,
        "position": _round_vec3(drone.position),
        "velocity": _round_vec3(drone.velocity),
        "heading": round(drone.heading, 3),
        "battery": round(drone.battery, 1),
        "status": drone.status.name.lower(),
        "sensor_active": drone.sensor_active,
        "comms_active": drone.comms_active,
        "current_task": drone.current_task,
        "target": (
            [round(drone.target.x, 1), round(drone.target.y, 1), round(drone.target.z, 1)]
            if drone.target
            else None
        ),
    }


def _serialize_survivor(survivor: Survivor) -> dict:
    return {
        "id": survivor.id,
        "position": [
            round(survivor.position.x, 1),
            round(survivor.position.y, 1),
            round(survivor.position.z, 1),
        ],
        "discovered": survivor.discovered,
        "discovered_by": survivor.discovered_by,
        "discovered_at_tick": survivor.discovered_at_tick,
        "mobile": survivor.mobile,
    }


def _serialize_event(event: SimEvent) -> dict:
    return {
        "type": event.type.name.lower(),
        "tick": event.tick,
        "drone_id": event.drone_id,
        "survivor_id": event.survivor_id,
    }


def _serialize_terrain(terrain: Terrain) -> dict:
    """Serialize terrain data. Heightmap sent as nested list for JSON compat."""
    return {
        "width": terrain.width,
        "height": terrain.height,
        "max_elevation": terrain.max_elevation,
        "heightmap": terrain.heightmap.tolist(),
        "biome_map": terrain.biome_map.tolist(),
    }


def _compress_fog_grid(fog_grid: np.ndarray) -> dict:
    """Compress fog grid using run-length encoding for efficient transmission.

    Returns a dict with width, height, and RLE-encoded data.
    Instead of sending a 256x256 array (65536 values) every tick,
    we send runs of identical values which is much smaller since
    fog changes incrementally.
    """
    flat = fog_grid.ravel()
    runs: list[list[int]] = []
    current_val = int(flat[0])
    count = 1

    for i in range(1, len(flat)):
        val = int(flat[i])
        if val == current_val:
            count += 1
        else:
            runs.append([current_val, count])
            current_val = val
            count = 1
    runs.append([current_val, count])

    return {
        "width": fog_grid.shape[1],
        "height": fog_grid.shape[0],
        "rle": runs,
    }
