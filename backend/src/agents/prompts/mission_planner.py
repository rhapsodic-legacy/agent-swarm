"""Prompt templates for the Claude-powered mission planner."""

from __future__ import annotations

SYSTEM_PROMPT = """You are the strategic mission planner for an autonomous drone swarm \
performing search-and-rescue over procedurally generated terrain.

Your role:
- Analyze fleet status, coverage progress, and terrain conditions
- Issue zone priorities and drone reassignments
- Adapt strategy based on human operator overrides
- Provide clear, actionable briefings

You make decisions every 10-30 seconds of simulated time based on significant events.

ALWAYS respond with a valid JSON object matching this schema:
{
    "zone_priorities": {"zone_id": "high|medium|low", ...},
    "reassignments": [{"drone_id": int, "action": "str", "target_zone": "str"}],
    "recall_drones": [int, ...],
    "reasoning": "Brief explanation of your strategic decision",
    "briefing": "One-sentence status update for the human operator"
}

Be concise. Prioritize unexplored high-value areas (urban, forest). Account for battery levels \
and drone failures. If a human override was given, respect it and explain how you're adapting."""


def build_mission_context(
    elapsed: float,
    total_drones: int,
    active_drones: int,
    failed_drones: list[int],
    coverage_pct: float,
    survivors_found: int,
    total_survivors: int,
    fleet_avg_battery: float,
    low_battery_drones: list[int],
    zone_coverage: dict[str, float],
    recent_events: list[str],
    human_overrides: list[str],
) -> str:
    """Build the user prompt with current mission state."""
    events_str = "\n".join(f"  - {e}" for e in recent_events[-10:]) if recent_events else "  None"
    overrides_str = (
        "\n".join(f"  - {o}" for o in human_overrides[-5:]) if human_overrides else "  None"
    )
    zones_str = "\n".join(f"  {zid}: {cov:.0f}% explored" for zid, cov in zone_coverage.items())

    return f"""Current mission state:

TIME: {elapsed:.0f}s elapsed
FLEET: {active_drones}/{total_drones} active, {len(failed_drones)} failed {failed_drones}
COVERAGE: {coverage_pct:.1f}% of terrain explored
SURVIVORS: {survivors_found}/{total_survivors} found
BATTERY: Fleet average {fleet_avg_battery:.0f}%, low battery: {low_battery_drones}

ZONE COVERAGE:
{zones_str}

RECENT EVENTS:
{events_str}

HUMAN OVERRIDES:
{overrides_str}

What is your strategic assessment and next directives?"""
