---
name: investigate
description: Systematic bug investigation — trace actual data through every layer instead of guessing. Use when something isn't working as expected.
user_invocable: true
---

# /investigate — Trace, Don't Guess

Use this when something doesn't work. DO NOT hypothesize or guess. Follow this exact procedure:

## Step 1: Reproduce with real data

Write a standalone Python script that exercises the exact code path and prints actual values:

```bash
cd /Users/jesseceresphoenix/Documents/agent_swarm/backend && uv run python -c "
# Import the modules under investigation
# Call the actual functions
# Print the actual return values, not what you expect
print(f'ACTUAL VALUE: {result}')
"
```

## Step 2: Trace the data through each layer

For a bug where X doesn't appear in the frontend:

1. **Source:** Where is X created? Print it right after creation.
2. **Storage:** Where is X stored? Print it from storage.
3. **Serialization:** What does the JSON look like? `json.dumps()` and print it.
4. **Wire:** What does the WebSocket actually send? Connect with `websockets` and print the raw message.
5. **Client:** What does the frontend receive? Add `console.log` at the receive point.
6. **Render:** What does the renderer do with it? Add `console.log` at the render call.

At each step, print ACTUAL VALUES. The bug is at the layer where the data disappears or changes.

## Step 3: Check the wire (WebSocket test)

```bash
uv run python -c "
import asyncio, json, websockets
async def check():
    async with websockets.connect('ws://127.0.0.1:PORT/ws', max_size=16*1024*1024) as ws:
        for _ in range(20):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
            if msg.get('type') == 'state_update':
                print('KEYS:', sorted(msg.keys()))
                print('FIELD_X:', msg.get('field_x', 'MISSING'))
                break
asyncio.run(check())
"
```

## Step 4: Frontend console check

Add temporary `console.log` at:
- The WebSocket receive handler
- The state callback
- The render function that uses the data

Look for: `undefined`, `[]`, wrong types, missing fields.

## Rules

- **Never say "it should work."** Prove it works with printed output.
- **Never assume the user made an error.** The bug is in the code.
- **Never skip a layer.** Trace from source to render, every step.
- **Fix only after you have evidence.** Not before.

## When to use

- User reports something doesn't work
- A feature you just built doesn't appear in the browser
- Tests pass but the visual result is wrong
- Any time you're about to type "I think the issue is..."
