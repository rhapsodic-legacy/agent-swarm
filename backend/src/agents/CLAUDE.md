# Agent Layer — LLM + Classical AI

## Architecture: Three-Tier Agent System

```
Tier 1: Mission Planner (Claude API)
    │   Runs: on significant events (survivor found, drone lost, coverage milestone)
    │   Scope: fleet-wide strategy, zone assignments, priority shifts
    │
Tier 2: Drone Reasoner (Mistral API, free tier)
    │   Runs: every N seconds per drone (async, non-blocking)
    │   Scope: per-drone tactical decisions (where to search, when to return)
    │
Tier 3: Reactive Controller (Classical AI)
        Runs: every tick (20Hz, synchronous)
        Scope: pathfinding, obstacle avoidance, formation keeping
```

## Critical Rules

1. **LLM calls NEVER block the sim loop.** They run as async tasks. Between LLM responses, drones use their last classical AI decision.
2. **Cache LLM responses** for identical state hashes. Don't re-query for the same situation.
3. **Rate limit Mistral calls** to stay within free tier. Batch drone decisions where possible.
4. **All LLM prompts live in `src/agents/prompts/`** as Jinja2 templates. Never inline prompts in code.
5. **Fallback gracefully**: If any LLM API is unavailable, fall back to classical AI. The system must work without any LLM calls.

## Classical AI Components

### Pathfinding (A*)
- Grid-based A* on terrain
- Cost function: base cost + elevation change penalty + biome penalty (water = impassable)
- Returns a list of waypoints, not just the destination

### Search Patterns
- `LawnmowerSearch`: Systematic parallel sweeps across assigned zone
- `SpiralSearch`: Outward spiral from a point of interest
- `FrontierSearch`: Move to nearest unexplored cell boundary
- `PrioritySearch`: Visit high-probability cells first (weighted by biome)

### Coordination
- Decentralized auction-based task allocation
- Each drone bids on available tasks based on distance + battery + current task priority
- Drones only coordinate with neighbors within communication range
- When a drone fails, its area is re-auctioned among nearby drones

## LLM Integration Patterns

### Mistral (Drone-Level)
```python
# Input context (kept small for free tier token limits)
{
    "drone_id": 3,
    "position": [120, 45, 340],
    "battery": 67,
    "nearby_drones": [1, 5],
    "local_explored_pct": 42,
    "survivors_found_nearby": 1,
    "current_task": "searching zone B",
    "recent_events": ["survivor found at (110, 330)", "drone 8 lost contact"]
}

# Expected output: tactical decision
{
    "action": "redirect_search",
    "target": [130, 350],
    "reasoning": "Survivor cluster likely near last find, redirecting northeast"
}
```

### Claude (Mission Planner)
```python
# Input context (richer, less frequent)
{
    "elapsed_time": 245,
    "total_drones": 20,
    "active_drones": 17,
    "failed_drones": [8, 12, 15],
    "coverage_pct": 38,
    "survivors_found": 4,
    "zones": {"A": 80, "B": 42, "C": 12, "D": 0},  # coverage per zone
    "fleet_avg_battery": 54,
    "human_overrides": ["prioritize zone C"]
}

# Expected output: strategic directive
{
    "zone_priorities": {"C": "high", "D": "high", "A": "low", "B": "medium"},
    "reassignments": {"drone_1": "zone_D", "drone_6": "zone_C"},
    "recall_low_battery": [9, 14],
    "reasoning": "Human prioritized zone C. Zone D is unexplored. Recalling low-battery units.",
    "briefing": "Redirecting assets to zones C and D per operator directive. Three drones recalled for recharge."
}
```

## Human Override Handling

When a human issues a command (via click, zone draw, or natural language):
1. Command is parsed into a `Command` object
2. If it conflicts with an agent decision, the human command wins
3. The override is recorded in the mission context
4. Next Claude mission planner call includes the override, so it adapts strategy
5. The system may ask for clarification if the override seems contradictory
