---
name: verify
description: End-to-end spatial verification for the drone swarm simulator. Run BEFORE any visual deploy to catch coordinate bugs, UV flips, stale configs, and data contract breakage.
user_invocable: true
---

# /verify — Pre-Deploy Verification

Run this skill before asking the user to open the browser. It catches the class of bugs that unit tests miss: coordinate system mismatches across backend→frontend, fog texture UV issues, spatial invariants, and data contract regressions.

## Steps

1. Run the backend E2E verification script:
```bash
cd /Users/jesseceresphoenix/Documents/agent_swarm/backend && uv run python scripts/verify_e2e.py
```

2. Run the backend unit tests:
```bash
cd /Users/jesseceresphoenix/Documents/agent_swarm/backend && uv run pytest tests/test_integration.py tests/test_serialization.py -x -q
```

3. Run the frontend type check:
```bash
cd /Users/jesseceresphoenix/Documents/agent_swarm/frontend && npx tsc --noEmit --skipLibCheck
```

4. Report results to the user. If ANY check fails, do NOT start the server or open the browser. Fix the issue first, then re-run /verify.

## What It Catches

- Fog clearing at wrong position (UV flip, axis transposition)
- Drones spawning away from base
- Zones assigned at world center instead of base
- Survivor cluster overlapping base
- Chunk serialization regressions (size, missing fields)
- Activity log not generating entries
- RLE compression roundtrip errors
- TypeScript compilation errors

## When to Run

- After ANY change to: fog rendering, drone positions, base position, terrain chunks, coordinate systems, search patterns
- Before telling the user to open the browser
- After a reset/restart if config changed
