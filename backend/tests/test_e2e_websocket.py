"""TRUE end-to-end WebSocket test.

Starts the actual server, connects via WebSocket, receives real messages,
and verifies the data the frontend would actually see. No mocks.

Run with: uv run pytest tests/test_e2e_websocket.py -x -v -s
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request

import pytest
import websockets

SERVER_PORT = 18765
SERVER_URL = f"ws://127.0.0.1:{SERVER_PORT}/ws"


@pytest.fixture(scope="module")
def server():
    """Start the actual backend server on a test port."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import uvicorn; from src.server.main import app; "
            f"uvicorn.run(app, host='127.0.0.1', port={SERVER_PORT}, "
            f"log_level='warning', ws_max_size=16*1024*1024)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for i in range(90):  # up to 90 seconds for chunk pre-generation
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=1)
            if resp.status == 200:
                print(f"\n[SERVER] Ready after {i + 1}s")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(f"Server did not start within 90 seconds.\nstderr: {stderr.decode()[-500:]}")

    yield proc

    proc.kill()
    proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_full_e2e_pipeline(server):
    """Connect to real server, receive messages, verify survivors exist.

    This test catches EXACTLY the bug the user reported: survivors not
    showing up in the frontend despite backend logs saying they exist.
    """
    chunk_messages: list[dict] = []
    state_updates: list[dict] = []
    overview = None
    mission_briefing = None

    async with websockets.connect(SERVER_URL, max_size=16 * 1024 * 1024) as ws:
        deadline = asyncio.get_event_loop().time() + 20.0
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg = json.loads(raw)
                msg_type = msg.get("type", "unknown")

                if msg_type == "world_overview":
                    overview = msg
                    print(
                        f"\n[OVERVIEW] world={msg['world_size']}m "
                        f"chunks={msg['chunks_x']}x{msg['chunks_z']}"
                    )

                elif msg_type == "chunk_terrain":
                    chunk_messages.append(msg)

                elif msg_type == "mission_briefing":
                    # Phase 2 scenario briefing has a "mission" key; the older
                    # planner-directive flavor doesn't.
                    if "mission" in msg:
                        mission_briefing = msg
                        m = msg["mission"]
                        print(
                            f"[OK] Mission: {m['title']} "
                            f"base=({m['base_position'][0]:.0f}, "
                            f"{m['base_position'][2]:.0f})"
                        )

                elif msg_type == "state_update":
                    state_updates.append(msg)
                    tick = msg.get("tick", 0)
                    if tick >= 30:
                        break

            except TimeoutError:
                continue

    # ======== ASSERTIONS ========
    print(f"\n{'=' * 60}")
    print("END-TO-END PIPELINE RESULTS")
    print(f"{'=' * 60}")

    # 1. World overview
    assert overview is not None, "No world_overview received"
    print(
        f"[OK] World overview: {overview['world_size']}m, "
        f"{overview['chunks_x']}x{overview['chunks_z']} chunks"
    )

    # 2. Chunks
    assert len(chunk_messages) > 0, "No chunks received"
    for c in chunk_messages[:3]:
        hm = base64.b64decode(c["heightmap_b64"])
        bm = base64.b64decode(c["biome_map_b64"])
        print(
            f"[OK] Chunk ({c['cx']},{c['cz']}): "
            f"hm={len(hm)}B bm={len(bm)}B survivors={c.get('survivor_count', '?')}"
        )
    if len(chunk_messages) > 3:
        print(f"     ... and {len(chunk_messages) - 3} more chunks")

    # 3. State updates
    assert len(state_updates) > 0, "No state updates received"
    last = state_updates[-1]

    # 4. Drones
    drones = last.get("drones", [])
    assert len(drones) > 0, "No drones in state"
    print(f"[OK] Drones: {len(drones)}")

    # 5. CRITICAL: all_survivors
    all_survivors = last.get("all_survivors", [])
    print(f"[{'OK' if all_survivors else 'FAIL'}] all_survivors: {len(all_survivors)}")
    assert len(all_survivors) > 0, (
        f"CRITICAL: all_survivors is EMPTY in state_update!\n"
        f"  This means god mode shows nothing.\n"
        f"  State keys: {sorted(last.keys())}\n"
        f"  world.survivors was empty — survivors not flowing from chunks to WorldState."
    )

    # 6. Survivor positions are valid numbers
    for s in all_survivors[:5]:
        pos = s.get("position", [])
        assert len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos), (
            f"Bad survivor position: {s}"
        )
        print(
            f"     Survivor {s['id']}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) "
            f"discovered={s.get('discovered')}"
        )

    # 7. Discovered survivors
    discovered = last.get("survivors", [])
    print(f"[OK] Discovered survivors: {len(discovered)}")

    # 8. Fog grid
    fog = last.get("fog_grid", {})
    assert "rle" in fog, "No fog grid in state"
    print(f"[OK] Fog grid: {fog.get('width')}x{fog.get('height')}")

    # 9. Drones spawn near the mission's base (now mission-driven, was hardcoded SW corner)
    assert mission_briefing is not None, (
        "No mission_briefing received — Phase 2 server should send one on connect"
    )
    base_x, _, base_z = mission_briefing["mission"]["base_position"]
    d0 = drones[0]
    pos = d0.get("position", [0, 0, 0])
    world_size = last.get("world_size", 10240)
    dist_from_base = ((pos[0] - base_x) ** 2 + (pos[2] - base_z) ** 2) ** 0.5
    print(
        f"[CHECK] Drone 0 at ({pos[0]:.0f}, {pos[2]:.0f}), "
        f"base=({base_x:.0f}, {base_z:.0f}), dist={dist_from_base:.0f}m"
    )
    assert dist_from_base < 2000, (
        f"Drones are {dist_from_base:.0f}m from mission base! Base position is wrong."
    )

    # 10. Survivors are NOT all clustered at the base
    base_survivors = [
        s
        for s in all_survivors
        if ((s["position"][0] - base_x) ** 2 + (s["position"][2] - base_z) ** 2) ** 0.5 < 1024
    ]
    remote_survivors = len(all_survivors) - len(base_survivors)
    print(f"[CHECK] Survivors near base: {len(base_survivors)}, remote: {remote_survivors}")
    assert remote_survivors > 0, "All survivors are at the base — they should be scattered away"

    # 11. All survivors must be within world bounds (0 to world_size)
    out_of_bounds = [
        s
        for s in all_survivors
        if s["position"][0] < 0
        or s["position"][0] > world_size
        or s["position"][2] < 0
        or s["position"][2] > world_size
    ]
    print(f"[CHECK] Out of world bounds: {len(out_of_bounds)}")
    assert len(out_of_bounds) == 0, (
        f"{len(out_of_bounds)} survivors outside world bounds! "
        f"First: {out_of_bounds[0] if out_of_bounds else 'N/A'}"
    )

    # 12. PoC grid should arrive at least once within 30 ticks
    poc_updates = [s for s in state_updates if s.get("poc_grid")]
    print(f"[CHECK] state_updates with poc_grid: {len(poc_updates)}")
    assert len(poc_updates) > 0, "No poc_grid in any state_update! Bayesian heatmap won't render."
    sample = poc_updates[-1]["poc_grid"]
    print(
        f"[CHECK] poc_grid size={sample['size']}, peak={sample['peak']:.5f}, "
        f"data_bytes={len(base64.b64decode(sample['data_b64']))}"
    )
    assert sample["size"] > 0
    assert sample["peak"] > 0
    # Data bytes should equal size * size
    data_bytes = base64.b64decode(sample["data_b64"])
    assert len(data_bytes) == sample["size"] * sample["size"], (
        f"PoC data size mismatch: {len(data_bytes)} vs {sample['size'] ** 2}"
    )

    # 9. Coverage
    coverage = last.get("coverage_pct", 0)
    print(f"[OK] Coverage: {coverage}%")

    print(f"\n{'=' * 60}")
    print(f"PASS: All {len(all_survivors)} survivors visible to frontend")
    print(f"{'=' * 60}")
