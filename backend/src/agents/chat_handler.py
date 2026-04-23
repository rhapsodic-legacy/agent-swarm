"""Chat command handler for natural language operator interaction.

Interprets human operator messages and converts them into simulation Commands
plus natural language responses. Uses Claude for interpretation with a keyword-based
fallback when the API is unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from src.agents.llm_client import call_claude
from src.simulation.types import Command, DroneStatus, Vec3, WorldState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the mission coordinator chat interface for an autonomous drone swarm \
performing search-and-rescue (SAR) over procedurally generated terrain.

You receive natural language messages from the human operator. Your job is to:
1. Interpret the operator's intent (query, command, or both).
2. Generate a helpful natural language reply — cite real numbers from the
   world state, never invent them.
3. Translate actionable requests into simulation commands.

ALWAYS respond with a valid JSON object matching this schema:
{
    "reply": "Human-readable response to the operator",
    "commands": [ ... ],
    "actions_taken": ["Redirected drone 3 to NE sector", ...]
}

Command types (pick the right shape for the operator's intent):

1. Point-and-move (a specific drone):
   {"type": "move_to", "drone_id": 3, "target": [x, z]}
   {"type": "return_to_base", "drone_id": 5}
   {"type": "hold_position", "drone_id": 7}

2. Area hint — the operator says "focus on the urban area" / "prioritize \
the eastern ridge". Emit a set_intel_pin: a point with radius that the \
swarm will bias toward. The market decides which drones redirect (it \
factors in distance, battery, current task). Don't try to assign drones \
yourself for area hints.
   {
     "type": "set_intel_pin",
     "pin_id": "valley_check",
     "position": [x, z],
     "radius": 400,
     "value": 1.0,
     "label": "valley south of crash site",
     "ttl_s": 300
   }
   Use ttl_s to auto-expire pins you added for a transient hint (30–600s).
   value 1.0 = normal urgency; bump to 2.0+ for "critical" / "drop everything".

3. Full polygon zone — only when the operator explicitly describes a \
region with clear boundaries ("the area between the river and the road").
   {
     "type": "zone_command",
     "zone_id": "east_ridge",
     "priority": "high",
     "polygon": [[x1,z1],[x2,z2],[x3,z3],[x4,z4]]
   }
   priority is "high" (5× pull), "low" (0.3× pull), or "avoid" (0× — \
never enter). Prefer intel pins over zones unless geometry matters.

4. Dismiss a hint the operator asked you to retract:
   {"type": "dismiss_intel_pin", "pin_id": "valley_check"}

Status queries ("give me a status", "why haven't we searched east?"):
return commands: [] and compose the reply from the world state given
below. Cite specific numbers (drone count per quadrant, battery, coverage).

Coordinate system:
- X axis: east (positive) / west (negative)
- Z axis: north (positive) / south (negative)
- Base position is given in the world state — use it as a reference
  anchor when the operator says "north of base" etc.
- Terrain bounds are given — don't emit coordinates outside them.

Rules:
- If the operator asks only a question, return commands: [].
- Keep replies concise, like a military ops center. Max 2 sentences.
- If the operator's request is ambiguous, ask for clarification and \
issue no commands.
- Never hallucinate drones, survivors, or positions. If you don't see \
it in the world state, say so."""


@dataclass
class ChatResponse:
    """Result of processing an operator chat message."""

    reply: str
    commands: list[Command]
    actions_taken: list[str]


@dataclass
class ChatHandler:
    """Interprets natural language operator commands for the drone swarm simulation.

    Uses Claude API for full natural language understanding, with a keyword-based
    fallback for basic commands when the API is unavailable.
    """

    interaction_log: list[dict] = field(default_factory=list)
    max_log_size: int = 100

    async def handle_message(self, message: str, world: WorldState) -> ChatResponse:
        """Process a human operator message and return a response with optional commands.

        Args:
            message: The operator's natural language message.
            world: Current simulation world state for context.

        Returns:
            ChatResponse with reply text, simulation commands, and action descriptions.
        """
        user_prompt = _build_user_prompt(message, world)

        llm_resp = await call_claude(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024,
            temperature=0.3,
        )

        if llm_resp.success and llm_resp.parsed:
            response = _parse_llm_response(llm_resp.parsed, world)
        else:
            logger.warning(
                "Claude unavailable for chat (%s), using keyword fallback",
                llm_resp.error or "no parsed response",
            )
            response = _fallback_parse(message, world)

        self._log_interaction(message, response, source=llm_resp.source)
        return response

    def _log_interaction(self, message: str, response: ChatResponse, source: str) -> None:
        """Record the interaction for debugging and audit."""
        entry = {
            "message": message,
            "reply": response.reply,
            "commands_count": len(response.commands),
            "actions": response.actions_taken,
            "source": source,
        }
        self.interaction_log.append(entry)
        if len(self.interaction_log) > self.max_log_size:
            self.interaction_log = self.interaction_log[-self.max_log_size :]
        logger.info(
            "Chat [%s]: %r -> %d commands, actions=%s",
            source,
            message[:80],
            len(response.commands),
            response.actions_taken,
        )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_user_prompt(message: str, world: WorldState) -> str:
    """Construct the user prompt with the operator message and world context."""
    terrain = world.terrain
    total_drones = len(world.drones)
    active = [d for d in world.drones if d.status == DroneStatus.ACTIVE]
    returning = [d for d in world.drones if d.status == DroneStatus.RETURNING]
    recharging = [d for d in world.drones if d.status == DroneStatus.RECHARGING]
    failed = [d for d in world.drones if d.status == DroneStatus.FAILED]
    survivors_found = sum(1 for s in world.survivors if s.discovered)
    total_survivors = len(world.survivors)

    # Coverage percentage from fog grid
    fog = np.asarray(world.fog_grid)
    total_cells = fog.size
    explored_cells = int(np.count_nonzero(fog != 0))
    coverage_pct = (explored_cells / total_cells * 100) if total_cells > 0 else 0.0

    # Drone details
    drone_lines: list[str] = []
    for d in world.drones:
        if d.status == DroneStatus.FAILED:
            continue
        drone_lines.append(
            f"  Drone {d.id}: pos=({d.position.x:.0f}, {d.position.z:.0f}) "
            f"battery={d.battery:.0f}% status={d.status.name} task={d.current_task}"
        )

    drones_str = "\n".join(drone_lines) if drone_lines else "  No active drones"

    # Recent events (last 5)
    recent_events = world.events[-5:] if world.events else ()
    events_lines: list[str] = []
    for ev in recent_events:
        detail = f"tick {ev.tick}: {ev.type.name}"
        if ev.drone_id is not None:
            detail += f" drone_id={ev.drone_id}"
        if ev.survivor_id is not None:
            detail += f" survivor_id={ev.survivor_id}"
        events_lines.append(f"  - {detail}")
    events_str = "\n".join(events_lines) if events_lines else "  None"

    elapsed_min = world.elapsed / 60
    elapsed_sec = world.elapsed % 60

    return f"""OPERATOR MESSAGE: {message}

--- WORLD STATE ---
Elapsed time: {elapsed_min:.0f}m {elapsed_sec:.0f}s ({world.elapsed:.0f}s total)
Terrain size: {terrain.width}m x {terrain.height}m
Base position: ({world.base_position.x:.0f}, {world.base_position.z:.0f})

Fleet summary:
  Total: {total_drones} | Active: {len(active)} | Returning: {len(returning)} \
| Recharging: {len(recharging)} | Failed: {len(failed)}
  Coverage: {coverage_pct:.1f}%
  Survivors found: {survivors_found}/{total_survivors}

Active drones:
{drones_str}

Recent events:
{events_str}

Respond with a JSON object containing "reply", "commands", and "actions_taken"."""


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _parse_llm_response(parsed: dict, world: WorldState) -> ChatResponse:
    """Convert the parsed JSON from Claude into a ChatResponse with proper Command objects."""
    reply = str(parsed.get("reply", "Acknowledged."))
    actions_taken = [str(a) for a in parsed.get("actions_taken", [])]
    raw_commands = parsed.get("commands", [])

    commands: list[Command] = []
    for raw in raw_commands:
        if not isinstance(raw, dict):
            continue
        cmd = _raw_command_to_command(raw, world)
        if cmd is not None:
            commands.append(cmd)

    return ChatResponse(reply=reply, commands=commands, actions_taken=actions_taken)


def _raw_command_to_command(raw: dict, world: WorldState) -> Command | None:
    """Convert a single raw JSON command dict into a Command, clamping coordinates."""
    cmd_type = raw.get("type", "")
    if not cmd_type:
        return None

    # --- Intel pins: point-based priority from natural language ---
    if cmd_type == "set_intel_pin":
        pos = raw.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return None
        x = _clamp(float(pos[0]), 0.0, float(world.terrain.width))
        z = _clamp(float(pos[1]), 0.0, float(world.terrain.height))
        data = {
            "action": "create",
            "pin_id": str(raw.get("pin_id", "llm_pin")),
            "position": [x, z],
            "radius": float(raw.get("radius", 400.0)),
            "value": float(raw.get("value", 1.0)),
            "label": str(raw.get("label", "")),
            "ttl_s": raw.get("ttl_s"),
        }
        return Command(type="set_intel_pin", zone_id=data["pin_id"], data=data)

    if cmd_type == "dismiss_intel_pin":
        pin_id = str(raw.get("pin_id", ""))
        if not pin_id:
            return None
        return Command(
            type="set_intel_pin",
            zone_id=pin_id,
            data={"action": "delete", "pin_id": pin_id},
        )

    # --- Zone polygons: explicit geometry ---
    if cmd_type == "zone_command":
        polygon = raw.get("polygon")
        priority = raw.get("priority")
        zone_id = raw.get("zone_id") or raw.get("zone")
        if not polygon or not priority or not zone_id:
            return None
        data = {
            "action": "create",
            "zone_id": str(zone_id),
            "priority": str(priority),
            "polygon": polygon,
        }
        return Command(
            type="set_priority",
            zone_id=str(zone_id),
            priority=str(priority),
            data=data,
        )

    # --- Direct drone commands (legacy) ---
    drone_id: int | None = raw.get("drone_id")
    if drone_id is not None:
        drone_id = int(drone_id)

    target: Vec3 | None = None
    raw_target = raw.get("target")
    if raw_target is not None and isinstance(raw_target, (list, tuple)) and len(raw_target) >= 2:
        x = float(raw_target[0])
        z = float(raw_target[1])
        x = _clamp(x, 0.0, float(world.terrain.width))
        z = _clamp(z, 0.0, float(world.terrain.height))
        target = Vec3(x, 0.0, z)

    zone_id_legacy: str | None = raw.get("zone")
    priority_legacy: str | None = raw.get("priority")

    return Command(
        type=cmd_type,
        drone_id=drone_id,
        target=target,
        zone_id=zone_id_legacy,
        priority=priority_legacy,
    )


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------


def _fallback_parse(message: str, world: WorldState) -> ChatResponse:
    """Parse simple keyword-based commands when Claude API is unavailable."""
    msg = message.strip().lower()

    # Status / report
    if msg in ("status", "report") or msg.startswith("status") or msg.startswith("report"):
        return _build_status_response(world)

    # Pause
    if msg == "pause":
        return ChatResponse(
            reply="Simulation paused.",
            commands=[],
            actions_taken=[],
        )

    # "drone N to X,Z" pattern
    match = re.match(r"drone\s+(\d+)\s+to\s+([\d.]+)\s*,\s*([\d.]+)", msg)
    if match:
        drone_id = int(match.group(1))
        x = _clamp(float(match.group(2)), 0.0, float(world.terrain.width))
        z = _clamp(float(match.group(3)), 0.0, float(world.terrain.height))
        target = Vec3(x, 0.0, z)
        cmd = Command(type="move_to", drone_id=drone_id, target=target)
        return ChatResponse(
            reply=f"Moving drone {drone_id} to ({x:.0f}, {z:.0f}).",
            commands=[cmd],
            actions_taken=[f"Move drone {drone_id} to ({x:.0f}, {z:.0f})"],
        )

    # Unrecognized
    return ChatResponse(
        reply=(
            "I can't process that without the AI system. Try: status, pause, or 'drone N to X,Z'"
        ),
        commands=[],
        actions_taken=[],
    )


def _build_status_response(world: WorldState) -> ChatResponse:
    """Generate a text status summary from the current world state."""
    active = sum(1 for d in world.drones if d.status == DroneStatus.ACTIVE)
    returning = sum(1 for d in world.drones if d.status == DroneStatus.RETURNING)
    recharging = sum(1 for d in world.drones if d.status == DroneStatus.RECHARGING)
    failed = sum(1 for d in world.drones if d.status == DroneStatus.FAILED)
    total = len(world.drones)

    fog = np.asarray(world.fog_grid)
    total_cells = fog.size
    explored = int(np.count_nonzero(fog != 0))
    coverage = (explored / total_cells * 100) if total_cells > 0 else 0.0

    survivors_found = sum(1 for s in world.survivors if s.discovered)
    total_survivors = len(world.survivors)

    avg_battery = sum(d.battery for d in world.drones) / max(total, 1)
    elapsed_min = int(world.elapsed // 60)
    elapsed_sec = int(world.elapsed % 60)

    reply = (
        f"Mission status at {elapsed_min}m {elapsed_sec}s:\n"
        f"  Fleet: {active} active, {returning} returning, "
        f"{recharging} recharging, {failed} failed ({total} total)\n"
        f"  Coverage: {coverage:.1f}%\n"
        f"  Survivors: {survivors_found}/{total_survivors} found\n"
        f"  Avg battery: {avg_battery:.0f}%"
    )
    return ChatResponse(reply=reply, commands=[], actions_taken=["Generated status report"])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))
