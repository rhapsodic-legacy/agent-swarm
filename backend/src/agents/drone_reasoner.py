"""Mistral-powered drone-level tactical reasoner.

Each drone can optionally get a Mistral-backed tactical decision on a slower
cadence (~every 5 seconds). Between LLM calls, the classical AI drives behavior.
Rate-limited to stay within Mistral's free tier.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from src.agents.llm_client import LLMResponse, call_mistral
from src.agents.prompts.drone_reasoner import SYSTEM_PROMPT, build_drone_context
from src.simulation.types import Drone, DroneStatus, WorldState

logger = logging.getLogger(__name__)


class DroneReasoner:
    """Tactical reasoner for individual drones, backed by Mistral API.

    Rate-limited: batches drone decisions and spaces API calls to stay
    within free tier limits (~1 call/second).
    """

    def __init__(self, max_concurrent: int = 2, call_interval: float = 2.0) -> None:
        self.max_concurrent = max_concurrent
        self.call_interval = call_interval  # seconds between batches
        self.last_call_time = 0.0
        self.pending_tasks: dict[int, asyncio.Task] = {}
        self.latest_decisions: dict[int, dict] = {}
        self.call_count = 0
        # Log of LLM decisions for debug panel
        self.decision_log: list[dict] = []
        # Track which drones had LLM decisions recently
        self._last_llm_tick: dict[int, int] = {}
        self._llm_interval_ticks = 100  # ~5 seconds at 20Hz

    def should_reason(self, drone: Drone, current_tick: int) -> bool:
        """Check if this drone should get an LLM decision."""
        if drone.status != DroneStatus.ACTIVE:
            return False
        last = self._last_llm_tick.get(drone.id, 0)
        return (current_tick - last) >= self._llm_interval_ticks

    def trigger_reasoning(self, drone: Drone, world: WorldState) -> None:
        """Start an async reasoning call for a specific drone."""
        # Check rate limiting
        now = time.monotonic()
        if now - self.last_call_time < self.call_interval:
            return
        if len(self.pending_tasks) >= self.max_concurrent:
            return

        # Don't re-trigger if already pending
        if drone.id in self.pending_tasks:
            task = self.pending_tasks[drone.id]
            if not task.done():
                return

        self.last_call_time = now
        self._last_llm_tick[drone.id] = world.tick
        context = self._build_context(drone, world)
        task = asyncio.create_task(self._call_reasoner(drone.id, context, world.tick))
        self.pending_tasks[drone.id] = task

    def consume_decisions(self) -> dict[int, dict]:
        """Collect all completed drone decisions. Non-blocking."""
        results: dict[int, dict] = {}
        done_ids: list[int] = []

        for drone_id, task in self.pending_tasks.items():
            if task.done():
                done_ids.append(drone_id)
                try:
                    response: LLMResponse = task.result()
                    if response.success and response.parsed:
                        decision = response.parsed
                        self.latest_decisions[drone_id] = decision
                        results[drone_id] = decision

                        self.decision_log.append(
                            {
                                "tick": self._last_llm_tick.get(drone_id, 0),
                                "drone_id": drone_id,
                                "source": response.source,
                                "latency_ms": response.latency_ms,
                                "action": decision.get("action", "unknown"),
                                "reasoning": decision.get("reasoning", ""),
                            }
                        )
                        if len(self.decision_log) > 100:
                            self.decision_log = self.decision_log[-100:]
                except Exception as e:
                    logger.warning("Drone %d reasoning failed: %s", drone_id, e)

        for did in done_ids:
            del self.pending_tasks[did]

        return results

    async def _call_reasoner(self, drone_id: int, context: str, tick: int) -> LLMResponse:
        """Make the actual LLM call."""
        self.call_count += 1
        logger.info("Drone %d reasoning call #%d (tick %d)", drone_id, self.call_count, tick)
        return await call_mistral(SYSTEM_PROMPT, context, max_tokens=256, temperature=0.3)

    def _build_context(self, drone: Drone, world: WorldState) -> str:
        """Build drone-local context from world state."""
        # Find nearby drones
        nearby: list[dict] = []
        for other in world.drones:
            if other.id == drone.id or other.status == DroneStatus.FAILED:
                continue
            dist = (drone.position - other.position).length_xz()
            if dist < drone.comms_range:
                nearby.append(
                    {
                        "id": other.id,
                        "x": other.position.x,
                        "z": other.position.z,
                        "battery": other.battery,
                        "task": other.current_task,
                    }
                )

        # Local explored percentage (50m radius around drone)
        local_pct = self._local_coverage(drone, world.fog_grid, radius=50)

        # Count nearby discovered survivors
        nearby_survivors = 0
        for s in world.survivors:
            if s.discovered:
                dist = (drone.position - s.position).length_xz()
                if dist < 80:
                    nearby_survivors += 1

        # Recent events relevant to this drone
        recent: list[str] = []
        for e in world.events:
            if e.drone_id == drone.id:
                recent.append(f"{e.type.name}")
            elif e.drone_id is not None:
                # Nearby drone events
                other = next((d for d in world.drones if d.id == e.drone_id), None)
                if other:
                    dist = (drone.position - other.position).length_xz()
                    if dist < drone.comms_range:
                        recent.append(f"Drone {e.drone_id}: {e.type.name}")

        return build_drone_context(
            drone_id=drone.id,
            position=(drone.position.x, drone.position.y, drone.position.z),
            battery=drone.battery,
            sensor_active=drone.sensor_active,
            nearby_drones=nearby,
            local_explored_pct=local_pct,
            survivors_found_nearby=nearby_survivors,
            current_task=drone.current_task,
            recent_events=recent,
        )

    def _local_coverage(self, drone: Drone, fog_grid: np.ndarray, radius: int = 50) -> float:
        """Calculate explored percentage in a radius around the drone."""
        h, w = fog_grid.shape
        cx = int(min(max(drone.position.x, 0), w - 1))
        cz = int(min(max(drone.position.z, 0), h - 1))
        r0 = max(0, cz - radius)
        r1 = min(h, cz + radius)
        c0 = max(0, cx - radius)
        c1 = min(w, cx + radius)
        region = fog_grid[r0:r1, c0:c1]
        total = region.size
        if total == 0:
            return 100.0
        explored = np.count_nonzero(region != 0)
        return (explored / total) * 100
