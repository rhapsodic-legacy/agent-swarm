# WebSocket Protocol Specification

## Connection

- URL: `ws://localhost:8765/ws`
- Protocol: JSON messages, one JSON object per WebSocket frame
- All messages have a `type` field

## Server → Client Messages

### `state_update`
Sent every simulation tick (~50ms at 20Hz).

```json
{
    "type": "state_update",
    "tick": 1234,
    "elapsed": 61.7,
    "terrain": {
        "heightmap": [[...], ...],
        "biome_map": [[...], ...],
        "width": 256,
        "height": 256,
        "max_elevation": 200
    },
    "drones": [
        {
            "id": 0,
            "position": [120.5, 45.0, 340.2],
            "velocity": [2.1, 0.0, -1.5],
            "heading": 1.57,
            "battery": 67.3,
            "status": "active",
            "sensor_active": true,
            "comms_active": true,
            "current_task": "searching",
            "target": [130.0, 350.0]
        }
    ],
    "survivors": [
        {
            "id": 0,
            "position": [110.0, 30.0, 330.0],
            "discovered": true,
            "discovered_by": 3,
            "discovered_at_tick": 1100
        }
    ],
    "fog_grid": [[0, 0, 1, 2, ...], ...],
    "comms_links": [[0, 1], [1, 5], [3, 5]],
    "events": [
        {"type": "survivor_found", "drone_id": 3, "survivor_id": 0, "tick": 1100}
    ]
}
```

**Note**: `terrain` is only sent on the first message and when terrain changes. Subsequent `state_update` messages omit it to save bandwidth. Client caches terrain locally.

**Fog grid values**: 0 = unexplored, 1 = explored (current), 2 = stale (explored > N ticks ago)

### `mission_briefing`
Two flavors share this message type:

**Scenario briefing** — sent on connect and after every reset. Describes the
active SAR scenario (lost hiker, plane crash, maritime, avalanche, disaster).

```json
{
    "type": "mission_briefing",
    "mission": {
        "name": "aircraft_crash",
        "title": "Aircraft Crash — Mountain SAR",
        "description": "Single-engine aircraft lost contact ...",
        "known_facts": [
            "Last radar contact at (3140, 2880)",
            "Heading 142°, debris cone 2400m long",
            "Survivor count unknown (1-8 estimated)"
        ],
        "base_position": [3500.0, 0.0, 3200.0],
        "survival_window_seconds": 14400.0,
        "intel_pins": [
            {"label": "Last radar contact", "kind": "radar", "position": [3140, 2880]},
            {"label": "Inferred impact area", "kind": "impact", "position": [4540, 4380]}
        ]
    },
    "available": ["aircraft_crash", "lost_hiker", "maritime_sar", "avalanche", "disaster_response"]
}
```

**Strategic directive** — sent when the Claude mission planner generates a
new strategic directive.

```json
{
    "type": "mission_briefing",
    "tick": 1234,
    "briefing": "Redirecting assets to zones C and D per operator directive.",
    "zone_priorities": {"A": "low", "B": "medium", "C": "high", "D": "high"},
    "reasoning": "Human prioritized zone C. Zone D is unexplored."
}
```

Disambiguate by presence of `mission` (scenario) vs `briefing` (directive).

### `evidence_discovered`
Sent when a drone finds a clue (footprint, debris, signal fire) during a
mission that has an evidence trail. The backend has already applied the
Bayesian posterior update to the PoC grid by the time this message is
sent — the frontend just needs to render the marker and refresh its
heatmap from the next `state_update`.

```json
{
    "type": "evidence_discovered",
    "tick": 1234,
    "id": 2,
    "kind": "footprint",
    "position": [3140.0, 45.0, 2880.0],
    "confidence": 0.6,
    "heading": 1.57,
    "age_hours": 1.5,
    "drone_id": 7
}
```

Kinds ship in Phase 3: `footprint` (directional), `debris` (ring),
`signal_fire` (high-confidence circle). Additional kinds may appear
later; the frontend should ignore unknown kinds rather than error.

### `agent_log`
Debug information about agent decisions.

```json
{
    "type": "agent_log",
    "tick": 1234,
    "drone_id": 3,
    "source": "mistral",
    "decision": "redirect_search",
    "reasoning": "Survivor cluster likely near last find"
}
```

## Client → Server Messages

### `command`
Human-issued command.

```json
{
    "type": "command",
    "command": "move_to",
    "drone_id": 3,
    "target": [150.0, 400.0]
}
```

### `zone_command`
Define or modify a search zone.

```json
{
    "type": "zone_command",
    "action": "create",
    "zone_id": "user_zone_1",
    "polygon": [[100, 200], [200, 200], [200, 400], [100, 400]],
    "priority": "high"
}
```

### `chat_message`
Natural language command for Claude to interpret.

```json
{
    "type": "chat_message",
    "message": "Focus search on the urban area to the north"
}
```

### `chat_response`
Server responds to chat messages.

```json
{
    "type": "chat_response",
    "message": "Understood. Redirecting 4 drones to the northern urban sector.",
    "actions_taken": ["reassigned drones 2, 5, 8, 11 to zone N"]
}
```

### `sim_control`
Simulation control commands.

```json
{
    "type": "sim_control",
    "action": "pause" | "resume" | "set_speed" | "reset",
    "value": 2.0,
    "config": {
        "mission": "lost_hiker",
        "drone_count": 20,
        "survivor_count": 25
    }
}
```

The `config` field is only consumed when `action` is `reset`. Setting
`config.mission` to a registered mission name swaps the active scenario;
clients receive a fresh `mission_briefing` on the next broadcast.
