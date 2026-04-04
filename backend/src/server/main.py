"""FastAPI WebSocket server for the Drone Swarm Simulator.

Runs the simulation loop in an async background task and broadcasts
state to all connected clients every tick.
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
from src.simulation.engine import create_world, get_coverage_pct, tick
from src.simulation.hazards import HazardSystem
from src.simulation.metrics import MetricsTracker
from src.simulation.replay import ReplayRecorder
from src.simulation.serializer import serialize_state
from src.simulation.types import Command, SimConfig, Vec3, WorldState
from src.simulation.weather import WeatherSystem

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
    terrain_size=512,
    drone_count=20,
    survivor_count=25,
    drone_sensor_range=40.0,
    drone_comms_range=120.0,
    drone_battery_drain_rate=0.12,
)
sim_running = True
sim_speed = 1.0  # multiplier
sim_reset_requested = False
chat_handler = ChatHandler()
# Shared reference to current world state (read by chat handler)
current_world: WorldState | None = None
# Pending reset config (set by UI, consumed by sim loop)
pending_reset_config: dict | None = None


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
        logger.info("Client disconnected (%d remaining)", len(clients))


async def broadcast(state_msg: dict, state_with_terrain: dict) -> None:
    """Send state to all clients. New clients get terrain data."""
    if not clients:
        return

    data_no_terrain = json.dumps(state_msg)
    data_with_terrain = json.dumps(state_with_terrain) if new_clients else None

    disconnected: list[WebSocket] = []
    for ws in clients:
        try:
            if ws in new_clients:
                await ws.send_text(data_with_terrain)
            else:
                await ws.send_text(data_no_terrain)
        except Exception:
            disconnected.append(ws)

    # Mark new clients as having received terrain
    new_clients.clear()

    for ws in disconnected:
        clients.discard(ws)


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
    global pending_commands, sim_reset_requested, current_world, pending_reset_config

    logger.info("Initializing simulation world...")
    world = create_world(sim_config)
    rng = np.random.default_rng(sim_config.terrain_seed)
    dt = 1.0 / sim_config.tick_rate
    coordinator = SwarmCoordinator(sim_config)
    weather = WeatherSystem(sim_config.terrain_seed, sim_config.terrain_size)
    hazards = HazardSystem(
        sim_config.terrain_size, sim_config.terrain_size, sim_config.terrain_seed
    )
    daycycle = DayCycle(day_length=300.0)
    metrics = MetricsTracker()
    replay = ReplayRecorder(record_interval=5)
    replay.start()

    logger.info(
        "Simulation ready: %d drones, %d survivors, %dx%d terrain",
        len(world.drones),
        len(world.survivors),
        world.terrain.width,
        world.terrain.height,
    )

    while True:
        loop_start = time.monotonic()

        # Handle reset
        if sim_reset_requested:
            sim_reset_requested = False
            new_seed = int(time.time()) % 100000

            # Apply user config overrides if provided
            cfg = pending_reset_config or {}
            pending_reset_config = None
            reset_config = SimConfig(
                terrain_size=int(cfg.get("terrain_size", sim_config.terrain_size)),
                terrain_seed=new_seed,
                max_elevation=float(cfg.get("max_elevation", sim_config.max_elevation)),
                drone_count=int(cfg.get("drone_count", sim_config.drone_count)),
                tick_rate=sim_config.tick_rate,
                survivor_count=int(cfg.get("survivor_count", sim_config.survivor_count)),
                drone_max_speed=float(cfg.get("drone_speed", sim_config.drone_max_speed)),
                drone_sensor_range=float(cfg.get("sensor_range", sim_config.drone_sensor_range)),
                drone_comms_range=float(cfg.get("comms_range", sim_config.drone_comms_range)),
                drone_battery_drain_rate=float(
                    cfg.get("battery_drain", sim_config.drone_battery_drain_rate)
                ),
                drone_battery_critical=sim_config.drone_battery_critical,
                sensor_failure_prob=sim_config.sensor_failure_prob,
                comms_failure_prob=sim_config.comms_failure_prob,
                fog_stale_ticks=sim_config.fog_stale_ticks,
                drone_cruise_altitude=sim_config.drone_cruise_altitude,
            )
            day_length = float(cfg.get("day_length", 300.0))
            world = create_world(reset_config)
            rng = np.random.default_rng(new_seed)
            coordinator = SwarmCoordinator(reset_config)
            weather = WeatherSystem(new_seed, reset_config.terrain_size)
            hazards = HazardSystem(reset_config.terrain_size, reset_config.terrain_size, new_seed)
            daycycle = DayCycle(day_length=day_length)
            metrics = MetricsTracker()
            replay = ReplayRecorder(record_interval=5)
            replay.start()
            pending_commands = []
            # All clients need fresh terrain
            new_clients.update(clients)
            logger.info("Simulation RESET (seed=%d)", new_seed)

        if sim_running:
            # Get AI agent commands
            agent_commands = coordinator.update(world, sim_config)

            # Merge with human commands (human overrides take priority)
            commands = list(pending_commands) + agent_commands
            pending_commands = []

            # Tick the simulation
            world = tick(world, dt * sim_speed, commands, rng=rng, config=sim_config)
            current_world = world

            # Update environmental systems
            weather.update(world.elapsed)
            daycycle.update(world.elapsed)
            metrics.record_tick(world)
            replay.record(world)

            # Log significant events
            for event in world.events:
                logger.info("Event [tick %d]: %s", world.tick, event.type.name)

            # Build agent + environment info for frontend
            agent_info = {
                "phase": coordinator.phase.name,
                "briefing": coordinator.mission_planner.latest_briefing,
                "planner_calls": coordinator.mission_planner.call_count,
                "reasoner_calls": coordinator.drone_reasoner.call_count,
                "weather": weather.serialize(),
                "daycycle": daycycle.serialize(),
                "metrics": metrics.serialize(),
                "hazards": hazards.serialize(),
            }

            # Prepare both versions of state
            state_msg = serialize_state(world, include_terrain=False, agent_info=agent_info)
            state_with_terrain = (
                serialize_state(world, include_terrain=True, agent_info=agent_info)
                if new_clients
                else state_msg
            )
            await broadcast(state_msg, state_with_terrain)

            # Periodic status log
            if world.tick % 200 == 0:
                coverage = get_coverage_pct(world.fog_grid)
                found = sum(1 for s in world.survivors if s.discovered)
                active = sum(1 for d in world.drones if d.status.name == "ACTIVE")
                logger.info(
                    "Tick %d | %.0fs | Coverage: %.1f%% | Found: %d/%d | Active: %d/%d",
                    world.tick,
                    world.elapsed,
                    coverage,
                    found,
                    len(world.survivors),
                    active,
                    len(world.drones),
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
