"""Prompt templates for the Mistral-powered drone-level reasoner."""

from __future__ import annotations

SYSTEM_PROMPT = """You are the tactical AI for a single drone in a search-and-rescue swarm. \
You make quick decisions about where to search next based on local information.

Your sensor range is 40m. You can only communicate with drones within 100m.

ALWAYS respond with a valid JSON object:
{
    "action": "search|investigate|reposition|return_to_base",
    "target": [x, z],
    "reasoning": "Brief one-sentence explanation"
}

Actions:
- search: Continue systematic search pattern toward target coordinates
- investigate: Move to a specific point of interest (e.g., near a found survivor, likely area)
- reposition: Move to reduce overlap with nearby drones
- return_to_base: Battery critical or mission complete

Be decisive. Prioritize unexplored areas. Avoid overlap with nearby drones. \
If battery is below 20%, return to base."""


def build_drone_context(
    drone_id: int,
    position: tuple[float, float, float],
    battery: float,
    sensor_active: bool,
    nearby_drones: list[dict],
    local_explored_pct: float,
    survivors_found_nearby: int,
    current_task: str,
    recent_events: list[str],
    mission_directive: str | None = None,
) -> str:
    """Build the user prompt with local drone state."""
    nearby_str = ""
    for nd in nearby_drones[:5]:
        nearby_str += (
            f"  Drone {nd['id']}: pos=({nd['x']:.0f},{nd['z']:.0f}), "
            f"battery={nd['battery']:.0f}%, task={nd['task']}\n"
        )
    if not nearby_str:
        nearby_str = "  None in range\n"

    events_str = "\n".join(f"  - {e}" for e in recent_events[-5:]) if recent_events else "  None"

    directive_str = f"\nMISSION DIRECTIVE: {mission_directive}" if mission_directive else ""

    return f"""Drone {drone_id} status:
POSITION: ({position[0]:.0f}, {position[2]:.0f})
BATTERY: {battery:.0f}%
SENSOR: {"active" if sensor_active else "OFFLINE"}
CURRENT TASK: {current_task}
LOCAL AREA: {local_explored_pct:.0f}% explored
SURVIVORS FOUND NEARBY: {survivors_found_nearby}

NEARBY DRONES:
{nearby_str}
RECENT EVENTS:
{events_str}{directive_str}

What should this drone do next?"""
