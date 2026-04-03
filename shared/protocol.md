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
Sent when the Claude mission planner generates a new strategic directive.

```json
{
    "type": "mission_briefing",
    "tick": 1234,
    "briefing": "Redirecting assets to zones C and D per operator directive.",
    "zone_priorities": {"A": "low", "B": "medium", "C": "high", "D": "high"},
    "reasoning": "Human prioritized zone C. Zone D is unexplored."
}
```

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
    "action": "pause" | "resume" | "set_speed",
    "value": 2.0
}
```
