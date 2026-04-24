"""FastAPI WebSocket server for the Drone Swarm Simulator.

Runs the simulation loop in an async background task and broadcasts
state to all connected clients every tick.

Supports both monolithic terrain (small maps) and chunked terrain (10km+ worlds).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.agents.chat_handler import ChatHandler
from src.agents.coordinator import SwarmCoordinator
from src.simulation.daycycle import DayCycle
from src.simulation.engine import tick_chunked
from src.simulation.hazards import HazardSystem
from src.simulation.metrics import MetricsTracker
from src.simulation.mission import (
    SearchMission,
    available_missions,
    build_mission,
)
from src.simulation.replay import ReplayRecorder
from src.simulation.types import (
    FOG_UNEXPLORED,
    ChunkedWorldConfig,
    Command,
    SimConfig,
    Terrain,
    Vec3,
    WorldState,
)
from src.simulation.weather import WeatherSystem
from src.terrain.chunked import ChunkedWorld

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    asyncio.create_task(simulation_loop())
    logger.info("Simulation loop started")
    yield


app = FastAPI(title="Drone Swarm Coordinator", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state ---
clients: set[WebSocket] = set()
new_clients: set[WebSocket] = set()  # clients that haven't received terrain yet
pending_commands: list[Command] = []
sim_config = SimConfig(
    terrain_size=1024,  # used for survivor density calculations in chunked mode
    drone_count=20,
    survivor_count=25,
    drone_sensor_range=40.0,
    drone_comms_range=120.0,
    drone_battery_drain_rate=0.02,  # %/s — ~83 min flight time, tuned for 10km world
    # Detection realism
    detection_requires_los=True,
    canopy_occlusion=0.7,
    urban_occlusion=0.5,
    weather_visibility=1.0,
    night_penalty=0.4,
    transponder_range=200.0,
    transponder_ratio=0.15,  # 15% of survivors have transponders
)
chunked_config = ChunkedWorldConfig(
    world_size=10240,
    chunk_size=1024,
    seed=42,
)
# Active mission — drives base position, ground-truth clusters, and the PoC prior.
# Defaults to aircraft_crash to preserve the existing scenario shape; can be
# overridden via the reset config's `mission` field.
current_mission_name: str = "aircraft_crash"
current_mission: SearchMission | None = None
sim_running = True
sim_speed = 1.0  # multiplier
sim_reset_requested = False
chat_handler = ChatHandler()
# Shared reference to current world state (read by chat handler)
current_world: WorldState | None = None
# Pending reset config (set by UI, consumed by sim loop)
pending_reset_config: dict | None = None
# Track which chunks each client has received
client_chunks: dict[WebSocket, set[tuple[int, int]]] = {}


def parse_command(msg: dict) -> Command | None:
    """Parse a WebSocket message into a Command."""
    msg_type = msg.get("type")

    if msg_type == "command":
        target = msg.get("target")
        return Command(
            type=msg.get("command", ""),
            drone_id=msg.get("drone_id"),
            target=Vec3(target[0], 0, target[1]) if target else None,
        )
    elif msg_type == "zone_command":
        return Command(
            type="set_priority",
            zone_id=msg.get("zone_id"),
            priority=msg.get("priority"),
            data=msg,
        )
    elif msg_type == "intel_pin_command":
        return Command(
            type="set_intel_pin",
            zone_id=msg.get("pin_id"),  # reuse zone_id slot for pin_id
            data=msg,
        )
    elif msg_type == "trust_command":
        return Command(type="set_trust", data=msg)
    elif msg_type == "sim_control":
        action = msg.get("action")
        global sim_running, sim_speed, sim_reset_requested
        if action == "pause":
            sim_running = False
        elif action == "resume":
            sim_running = True
        elif action == "set_speed":
            sim_speed = float(msg.get("value", 1.0))
        elif action == "reset":
            global pending_reset_config
            sim_reset_requested = True
            sim_running = True
            # Store any config overrides sent with reset
            config_data = msg.get("config")
            pending_reset_config = config_data if isinstance(config_data, dict) else None
        return None
    return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.add(websocket)
    new_clients.add(websocket)
    client_chunks[websocket] = set()
    logger.info("Client connected (%d total)", len(clients))

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle chat messages async
            if message.get("type") == "chat_message":
                asyncio.create_task(_handle_chat(websocket, message.get("message", "")))
                continue

            cmd = parse_command(message)
            if cmd is not None:
                pending_commands.append(cmd)
                logger.info("Command queued: %s", cmd.type)
    except WebSocketDisconnect:
        clients.discard(websocket)
        new_clients.discard(websocket)
        client_chunks.pop(websocket, None)
        logger.info("Client disconnected (%d remaining)", len(clients))


async def broadcast_chunked(
    state_msg: dict,
    chunked_world: ChunkedWorld,
    active_chunks: list,
    world: WorldState,
    agent_info: dict,
) -> None:
    """Send state to all clients with chunk-based terrain."""
    if not clients:
        return

    disconnected: list[WebSocket] = []

    for ws in list(clients):
        try:
            chunks_to_send: list[dict] = []
            ws_chunks = client_chunks.get(ws, set())

            if ws in new_clients:
                # New client — send world overview + starting chunks
                overview = chunked_world.serialize_overview()
                overview["type"] = "world_overview"
                await ws.send_text(json.dumps(overview))

                # Send mission briefing so the operator sees scenario context
                if current_mission is not None:
                    briefing = {
                        "type": "mission_briefing",
                        "mission": current_mission.to_briefing_dict(),
                        "available": available_missions(),
                    }
                    await ws.send_text(json.dumps(briefing))

                # Send chunks near the base (where drones start)
                base = world.base_position
                nearby = chunked_world.get_chunks_near(base.x, base.z, 2048.0)
                for chunk in nearby:
                    key = (chunk.coord.cx, chunk.coord.cz)
                    chunk_data = chunked_world.serialize_chunk(chunk.coord)
                    chunk_data["type"] = "chunk_terrain"
                    # Include fog for this chunk
                    chunk_data["fog_rle"] = _compress_chunk_fog(chunk.fog_grid)
                    chunks_to_send.append(chunk_data)
                    ws_chunks.add(key)
            else:
                # Send new chunks that drones have entered
                for chunk in active_chunks:
                    key = (chunk.coord.cx, chunk.coord.cz)
                    if key not in ws_chunks:
                        chunk_data = chunked_world.serialize_chunk(chunk.coord)
                        chunk_data["type"] = "chunk_terrain"
                        chunk_data["fog_rle"] = _compress_chunk_fog(chunk.fog_grid)
                        chunks_to_send.append(chunk_data)
                        ws_chunks.add(key)

            client_chunks[ws] = ws_chunks

            # Send any new chunks
            for cd in chunks_to_send:
                await ws.send_text(json.dumps(cd))

            # Send the per-tick state (drones, fog updates for active chunks, etc.)
            fog_updates = {}
            for chunk in active_chunks:
                key = f"{chunk.coord.cx},{chunk.coord.cz}"
                fog_updates[key] = _compress_chunk_fog(chunk.fog_grid)

            state_msg["chunk_fog_updates"] = fog_updates
            await ws.send_text(json.dumps(state_msg))

        except Exception:
            disconnected.append(ws)

    new_clients.clear()
    for ws in disconnected:
        clients.discard(ws)
        client_chunks.pop(ws, None)


def _compress_chunk_fog(fog_grid: np.ndarray) -> dict:
    """RLE compress a chunk's fog grid."""
    flat = fog_grid.ravel()
    if len(flat) == 0:
        return {"width": 0, "height": 0, "rle": []}
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
    return {"width": fog_grid.shape[1], "height": fog_grid.shape[0], "rle": runs}


async def _handle_chat(websocket: WebSocket, message: str) -> None:
    """Handle a chat message from the operator."""
    world = current_world
    if world is None:
        reply = {"type": "chat_response", "message": "Simulation not ready yet."}
        await websocket.send_text(json.dumps(reply))
        return

    logger.info("Chat: %s", message)
    response = await chat_handler.handle_message(message, world)

    # Queue any commands from the chat response
    pending_commands.extend(response.commands)

    # Send reply to the client
    reply = {
        "type": "chat_response",
        "message": response.reply,
        "actions_taken": response.actions_taken,
    }
    with contextlib.suppress(Exception):
        await websocket.send_text(json.dumps(reply))


async def simulation_loop() -> None:
    """Main simulation loop — runs as a background async task."""
    global pending_commands, sim_reset_requested, current_world, pending_reset_config, sim_config
    global current_mission, current_mission_name

    logger.info("Initializing chunked world (%dm, %dm chunks)...",
                chunked_config.world_size, chunked_config.chunk_size)

    seed = chunked_config.seed
    world_size = chunked_config.world_size

    # Build the initial mission. Drives clusters, base position, and PoC prior.
    mission = build_mission(current_mission_name, world_size, seed)
    current_mission = mission
    logger.info("Mission: %s (base=%.0f,%.0f)",
                mission.title, mission.base_position.x, mission.base_position.z)

    chunked_world = ChunkedWorld(
        chunked_config.world_size,
        chunked_config.chunk_size,
        seed,
        sim_config,
        clusters=mission.clusters,
    )

    base = mission.base_position

    from src.simulation.drone import create_drone_fleet
    drones = create_drone_fleet(sim_config.drone_count, base, sim_config)
    # Terrain stub with lazy proxies so agent code can do heightmap[row][col]
    stub_terrain = Terrain(
        width=world_size,
        height=world_size,
        max_elevation=sim_config.max_elevation,
        heightmap=chunked_world.make_heightmap_proxy(),
        biome_map=chunked_world.make_biome_proxy(),
        survivors=(),
        seed=seed,
    )
    # Coarse global fog grid for agent AI (1 cell ≈ 10m → 1024x1024 for 10km world)
    fog_res = max(world_size // 10, 256)
    global_fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)

    # Bayesian search map — seeded by the active mission. Phase 3 will widen
    # the gap between this prior and the ground-truth clusters; for now the
    # prior is shaped from the same cluster geometry the mission generated.
    from src.simulation.search_map import SearchMap
    search_map = SearchMap.empty(world_size=float(world_size), cell_size=40.0)
    mission.seed_poc_grid(search_map)

    world = WorldState(
        tick=0,
        elapsed=0.0,
        terrain=stub_terrain,
        drones=drones,
        survivors=(),
        fog_grid=global_fog,
        comms_links=(),
        events=(),
        base_position=base,
        tick_rate=sim_config.tick_rate,
        search_map=search_map,
        evidence=tuple(mission.evidence),
    )

    fog_scale = world_size / fog_res  # meters per fog cell
    rng = np.random.default_rng(seed)
    dt = 1.0 / sim_config.tick_rate
    coordinator = SwarmCoordinator(sim_config)
    weather = WeatherSystem(seed, world_size)
    hazards = HazardSystem(world_size, world_size, seed)
    daycycle = DayCycle(day_length=300.0)
    metrics = MetricsTracker(mission=mission)
    replay = ReplayRecorder(record_interval=5)
    replay.start()

    # Pre-generate only the 4 chunks directly around the base (not 18)
    logger.info("Pre-generating chunks near base...")
    base_chunks = chunked_world.get_chunks_near(base.x, base.z, 1200.0)
    logger.info("Pre-generated %d chunks near base", len(base_chunks))

    logger.info(
        "Chunked world ready: %d drones, %dx%d world, %dx%d chunks (%d total)",
        len(world.drones),
        world_size, world_size,
        chunked_config.chunk_size, chunked_config.chunk_size,
        chunked_world.get_total_chunks(),
    )

    while True:
        loop_start = time.monotonic()

        # Handle reset
        if sim_reset_requested:
            sim_reset_requested = False
            new_seed = int(time.time()) % 100000

            cfg = pending_reset_config or {}
            pending_reset_config = None

            # Apply detection config overrides
            sim_config = SimConfig(
                terrain_size=sim_config.terrain_size,
                drone_count=int(cfg.get("drone_count", sim_config.drone_count)),
                survivor_count=int(cfg.get("survivor_count", sim_config.survivor_count)),
                drone_max_speed=float(cfg.get("drone_speed", sim_config.drone_max_speed)),
                drone_sensor_range=float(cfg.get("sensor_range", sim_config.drone_sensor_range)),
                drone_comms_range=float(cfg.get("comms_range", sim_config.drone_comms_range)),
                drone_battery_drain_rate=float(cfg.get("battery_drain", sim_config.drone_battery_drain_rate)),
                max_elevation=float(cfg.get("max_elevation", sim_config.max_elevation)),
                canopy_occlusion=float(cfg.get("canopy_occlusion", sim_config.canopy_occlusion)),
                urban_occlusion=float(cfg.get("urban_occlusion", sim_config.urban_occlusion)),
                weather_visibility=float(cfg.get("weather_visibility", sim_config.weather_visibility)),
                night_penalty=float(cfg.get("night_penalty", sim_config.night_penalty)),
                transponder_ratio=float(cfg.get("transponder_ratio", sim_config.transponder_ratio)),
                detection_requires_los=sim_config.detection_requires_los,
                transponder_range=sim_config.transponder_range,
            )

            # Pick mission for the reset (defaults to current).
            mission_name = str(cfg.get("mission", current_mission_name))
            if mission_name not in available_missions():
                logger.warning("Unknown mission '%s', falling back to %s",
                               mission_name, current_mission_name)
                mission_name = current_mission_name
            current_mission_name = mission_name
            mission = build_mission(mission_name, world_size, new_seed)
            current_mission = mission
            logger.info("Reset mission: %s", mission.title)

            # Rebuild chunked world with the new mission's clusters
            chunked_world = ChunkedWorld(
                chunked_config.world_size,
                chunked_config.chunk_size,
                new_seed,
                sim_config,
                clusters=mission.clusters,
            )

            base = mission.base_position
            drones = create_drone_fleet(
                int(cfg.get("drone_count", sim_config.drone_count)),
                base, sim_config,
            )
            stub_terrain = Terrain(
                width=world_size,
                height=world_size,
                max_elevation=sim_config.max_elevation,
                heightmap=chunked_world.make_heightmap_proxy(),
                biome_map=chunked_world.make_biome_proxy(),
                survivors=(),
                seed=new_seed,
            )
            global_fog = np.full((fog_res, fog_res), FOG_UNEXPLORED, dtype=np.int8)
            # Reset search map seeded by the new mission's prior
            search_map = SearchMap.empty(world_size=float(world_size), cell_size=40.0)
            mission.seed_poc_grid(search_map)
            world = WorldState(
                tick=0,
                elapsed=0.0,
                terrain=stub_terrain,
                drones=drones,
                survivors=(),
                fog_grid=global_fog,
                comms_links=(),
                events=(),
                base_position=base,
                tick_rate=sim_config.tick_rate,
                search_map=search_map,
                evidence=tuple(mission.evidence),
            )
            rng = np.random.default_rng(new_seed)
            day_length = float(cfg.get("day_length", 300.0))
            coordinator = SwarmCoordinator(sim_config)
            weather = WeatherSystem(new_seed, world_size)
            hazards = HazardSystem(world_size, world_size, new_seed)
            daycycle = DayCycle(day_length=day_length)
            metrics = MetricsTracker(mission=mission)
            replay = ReplayRecorder(record_interval=5)
            replay.start()
            pending_commands = []

            # Clear all client chunk caches so they get fresh terrain
            for ws in client_chunks:
                client_chunks[ws] = set()
            new_clients.update(clients)
            logger.info("Simulation RESET (seed=%d)", new_seed)

        if sim_running:
            # Yield to event loop before heavy work (prevents startup blocking)
            await asyncio.sleep(0)

            # Drain the pending-command queue atomically so a concurrent
            # websocket push can't lose commands between iteration + reset.
            queued = list(pending_commands)
            pending_commands = []

            # Zone commands (set_priority) and intel pins (set_intel_pin) are
            # meta-commands that reshape the coordinator's target scoring —
            # apply them before it builds this tick's commands so the bias
            # takes effect immediately. Everything else falls through to the
            # drone-level command path.
            drone_commands: list[Command] = []
            for cmd in queued:
                if cmd.type == "set_priority":
                    coordinator.apply_zone_command(cmd, world)
                elif cmd.type == "set_intel_pin":
                    coordinator.apply_intel_pin_command(cmd, world)
                elif cmd.type == "set_trust":
                    data = cmd.data or {}
                    value = data.get("value")
                    if value is not None:
                        coordinator.adaptive.set_trust(float(value))
                else:
                    # Human-issued drone commands (move_to, RTB, hold).
                    # Record as overrides so the adaptive layer can learn
                    # that the drone's prior task wasn't what the operator
                    # wanted.
                    if cmd.drone_id is not None:
                        agent = coordinator.agents.get(cmd.drone_id)
                        if agent is not None:
                            coordinator.adaptive.record_operator_override(
                                world.tick, agent.task.name,
                            )
                    drone_commands.append(cmd)

            # Get AI agent commands
            agent_commands = coordinator.update(world, sim_config)

            # Human commands apply LAST so they win within the tick —
            # _apply_command iterates in order and later writes overwrite
            # earlier ones. Documented contract: human overrides agent.
            commands = agent_commands + drone_commands

            # Update weather BEFORE the tick so this tick's physics sees
            # the same wind that gets broadcast alongside the state snapshot.
            weather.update(world.elapsed)

            # Tick with chunked terrain (wind_fn queries live weather)
            world = tick_chunked(
                world, dt * sim_speed, chunked_world, commands,
                rng=rng, config=sim_config, wind_fn=weather.get_wind_at,
            )
            current_world = world

            # Update remaining environmental systems
            daycycle.update(world.elapsed)
            metrics.record_tick(world)
            replay.record(world)

            # Log significant events
            for event in world.events:
                logger.info("Event [tick %d]: %s", world.tick, event.type.name)

            # Get active chunks for broadcasting
            drone_positions = [d.position for d in world.drones]
            active_chunks = chunked_world.get_active_chunks(drone_positions)

            # Compute coverage across all generated chunks
            total_cells = 0
            explored_cells = 0
            for coord, chunk in chunked_world._cache.items():
                total_cells += chunk.fog_grid.size
                explored_cells += int(np.count_nonzero(chunk.fog_grid != FOG_UNEXPLORED))
            coverage_pct = (explored_cells / max(total_cells, 1)) * 100.0

            # Build agent + environment info for frontend
            # Send activity log entries from the last 10 ticks (avoid flooding)
            recent_log = coordinator.get_recent_log(since_tick=max(0, world.tick - 10))
            agent_info = {
                "phase": coordinator.phase.name,
                "briefing": coordinator.mission_planner.latest_briefing,
                "planner_calls": coordinator.mission_planner.call_count,
                "reasoner_calls": coordinator.drone_reasoner.call_count,
                "weather": weather.serialize(),
                "daycycle": daycycle.serialize(),
                "metrics": metrics.serialize(),
                "hazards": hazards.serialize(),
                "activity_log": recent_log,
            }

            # State message (no terrain — chunks sent separately)
            found_survivors = [s for s in world.survivors if s.discovered]
            state_msg = {
                "type": "state_update",
                "tick": world.tick,
                "elapsed": round(world.elapsed, 2),
                "drones": [
                    {
                        "id": d.id,
                        "position": [round(d.position.x, 1), round(d.position.y, 1), round(d.position.z, 1)],
                        "velocity": [round(d.velocity.x, 1), round(d.velocity.y, 1), round(d.velocity.z, 1)],
                        "heading": round(d.heading, 3),
                        "battery": round(d.battery, 1),
                        "status": d.status.name.lower(),
                        "sensor_active": d.sensor_active,
                        "comms_active": d.comms_active,
                        "current_task": d.current_task,
                        "target": (
                            [round(d.target.x, 1), round(d.target.y, 1), round(d.target.z, 1)]
                            if d.target else None
                        ),
                    }
                    for d in world.drones
                ],
                "survivors": [
                    {
                        "id": s.id,
                        "position": [round(s.position.x, 1), round(s.position.y, 1), round(s.position.z, 1)],
                        "discovered": s.discovered,
                        "discovered_by": s.discovered_by,
                        "discovered_at_tick": s.discovered_at_tick,
                        "mobile": s.mobile,
                    }
                    for s in found_survivors
                ],
                "all_survivors": [
                    {
                        "id": s.id,
                        "position": [round(s.position.x, 1), round(s.position.y, 1), round(s.position.z, 1)],
                        "discovered": s.discovered,
                        "mobile": s.mobile,
                    }
                    for s in chunked_world.get_all_survivors()
                ],
                "fog_grid": _compress_chunk_fog(world.fog_grid),
                "comms_links": [list(link) for link in world.comms_links],
                "events": [
                    {"type": e.type.name.lower(), "tick": e.tick, "drone_id": e.drone_id, "survivor_id": e.survivor_id}
                    for e in world.events
                ],
                "evidence": [
                    {
                        "id": ev.id,
                        "kind": ev.kind,
                        "position": [round(ev.position.x, 1), round(ev.position.y, 1), round(ev.position.z, 1)],
                        "confidence": round(ev.confidence, 2),
                        "heading": round(ev.heading, 3) if ev.heading is not None else None,
                        "age_hours": ev.age_hours,
                        "discovered_by": ev.discovered_by,
                        "discovered_at_tick": ev.discovered_at_tick,
                    }
                    for ev in world.evidence
                    if ev.discovered
                ],
                "coverage_pct": round(coverage_pct, 1),
                "agent_info": agent_info,
                "world_size": world_size,
                "chunk_size": chunked_config.chunk_size,
                "zones": coordinator.serialize_zones(),
                "intel_pins": coordinator.serialize_intel_pins(),
                "adaptive": coordinator.adaptive.summary(),
            }

            # One-shot notifications for each clue discovered this tick —
            # frontend uses these to pulse the heatmap / log the find.
            newly_found_evidence = [
                e for e in world.events
                if e.type.name == "EVIDENCE_FOUND"
            ]

            # PoC heatmap: send every 10 ticks (1 Hz at 10 Hz sim).
            # 64x64 grid at uint8 = 4KB per broadcast (cheap).
            if world.tick % 10 == 0 and world.search_map is not None:
                import base64

                from src.simulation.search_map import SearchMap
                sm: SearchMap = world.search_map  # type: ignore[assignment]
                down = sm.downsample(64)
                peak = max(float(down.max()), 1e-9)
                # Quantize to uint8 relative to current peak (preserves
                # relative hotspot visibility even as absolute mass decreases)
                arr = np.clip((down / peak) * 255.0, 0, 255).astype(np.uint8)
                state_msg["poc_grid"] = {
                    "size": 64,
                    "world_size": world_size,
                    "peak": peak,
                    "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
                }

            await broadcast_chunked(state_msg, chunked_world, active_chunks, world, agent_info)

            # Broadcast per-discovery evidence notifications. Look up the
            # full Evidence record for each EVIDENCE_FOUND event so the
            # frontend gets geometry + heading, not just the event stub.
            if newly_found_evidence:
                evidence_by_id = {e.id: e for e in world.evidence}
                for event in newly_found_evidence:
                    data = event.data or {}
                    eid = data.get("evidence_id")
                    ev = evidence_by_id.get(eid) if eid is not None else None
                    if ev is None:
                        continue
                    msg = {
                        "type": "evidence_discovered",
                        "tick": event.tick,
                        "id": ev.id,
                        "kind": ev.kind,
                        "position": [
                            round(ev.position.x, 1),
                            round(ev.position.y, 1),
                            round(ev.position.z, 1),
                        ],
                        "confidence": round(ev.confidence, 2),
                        "heading": round(ev.heading, 3) if ev.heading is not None else None,
                        "age_hours": ev.age_hours,
                        "drone_id": event.drone_id,
                    }
                    payload = json.dumps(msg)
                    disconnected: list[WebSocket] = []
                    for ws in list(clients):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            disconnected.append(ws)
                    for ws in disconnected:
                        clients.discard(ws)
                        client_chunks.pop(ws, None)

            # Evict stale chunks periodically
            if world.tick % 100 == 0:
                evicted = chunked_world.evict_stale(max_age_seconds=120.0)
                if evicted > 0:
                    logger.info("Evicted %d stale chunks", evicted)

            # Periodic status log
            if world.tick % 200 == 0:
                found = sum(1 for s in world.survivors if s.discovered)
                active = sum(1 for d in world.drones if d.status.name == "ACTIVE")
                logger.info(
                    "Tick %d | %.0fs | Coverage: %.1f%% | Found: %d | Active: %d/%d | Chunks: %d",
                    world.tick,
                    world.elapsed,
                    coverage_pct,
                    found,
                    active,
                    len(world.drones),
                    chunked_world.get_generated_count(),
                )

        # Sleep to maintain tick rate
        elapsed = time.monotonic() - loop_start
        sleep_time = max(0, dt - elapsed)
        await asyncio.sleep(sleep_time)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "clients": len(clients), "sim_running": sim_running}


def main() -> None:
    import uvicorn

    logger.info("Starting Drone Swarm Server on port 8765")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
        ws_max_size=16 * 1024 * 1024,  # 16MB for large terrain payloads
    )


if __name__ == "__main__":
    main()
