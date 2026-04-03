"""Claude-powered mission planner.

Runs asynchronously on significant events (survivor found, drone lost, coverage milestones).
Provides strategic directives that the SwarmCoordinator incorporates into its decision-making.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from src.agents.llm_client import LLMResponse, call_claude
from src.agents.prompts.mission_planner import SYSTEM_PROMPT, build_mission_context
from src.simulation.types import DroneStatus, EventType, WorldState

logger = logging.getLogger(__name__)


class MissionPlanner:
    """Strategic mission planner backed by Claude API.

    Runs asynchronously — never blocks the simulation loop.
    Falls back to a no-op if the API is unavailable.
    """

    def __init__(self) -> None:
        self.last_call_tick = 0
        self.min_interval_ticks = 200  # ~10 seconds at 20Hz
        self.latest_directive: dict | None = None
        self.latest_briefing: str = ""
        self.latest_reasoning: str = ""
        self.call_count = 0
        self.pending_task: asyncio.Task | None = None
        self.human_overrides: list[str] = []
        self.event_log: list[str] = []
        # Log of all decisions for frontend debug panel
        self.decision_log: list[dict] = []

    def record_event(self, event_type: str, detail: str) -> None:
        """Record a significant event for the next planner call."""
        self.event_log.append(f"[{event_type}] {detail}")
        # Keep last 20
        if len(self.event_log) > 20:
            self.event_log = self.event_log[-20:]

    def record_human_override(self, override: str) -> None:
        """Record a human override command."""
        self.human_overrides.append(override)
        if len(self.human_overrides) > 10:
            self.human_overrides = self.human_overrides[-10:]

    def should_plan(self, world: WorldState) -> bool:
        """Check if we should trigger a new planning cycle."""
        if self.pending_task and not self.pending_task.done():
            return False  # Still waiting for last response

        tick_delta = world.tick - self.last_call_tick
        if tick_delta < self.min_interval_ticks:
            return False

        # Trigger on significant events this tick
        for event in world.events:
            if event.type in (
                EventType.SURVIVOR_FOUND,
                EventType.DRONE_FAILED,
                EventType.DRONE_BATTERY_CRITICAL,
            ):
                return True

        # Trigger on coverage milestones (every 10%)
        coverage = self._get_coverage(world.fog_grid)
        last_coverage = getattr(self, "_last_coverage", 0.0)
        if int(coverage / 10) > int(last_coverage / 10):
            self._last_coverage = coverage
            return True

        # Periodic planning every min_interval_ticks
        return tick_delta >= self.min_interval_ticks

    def trigger_plan(self, world: WorldState) -> None:
        """Start an async planning call. Non-blocking."""
        self.last_call_tick = world.tick
        context = self._build_context(world)
        self.pending_task = asyncio.create_task(self._call_planner(context, world.tick))

    def consume_result(self) -> dict | None:
        """Check if a planning result is ready. Non-blocking."""
        if self.pending_task and self.pending_task.done():
            try:
                result = self.pending_task.result()
                self.pending_task = None
                if result and result.success and result.parsed:
                    self.latest_directive = result.parsed
                    self.latest_briefing = result.parsed.get("briefing", "")
                    self.latest_reasoning = result.parsed.get("reasoning", "")
                    self.decision_log.append(
                        {
                            "tick": self.last_call_tick,
                            "source": result.source,
                            "latency_ms": result.latency_ms,
                            "briefing": self.latest_briefing,
                            "reasoning": self.latest_reasoning,
                        }
                    )
                    if len(self.decision_log) > 50:
                        self.decision_log = self.decision_log[-50:]
                    return self.latest_directive
            except Exception as e:
                logger.warning("Mission planner task failed: %s", e)
                self.pending_task = None
        return None

    async def _call_planner(self, context: str, tick: int) -> LLMResponse:
        """Make the actual LLM call."""
        self.call_count += 1
        logger.info("Mission planner call #%d (tick %d)", self.call_count, tick)
        return await call_claude(SYSTEM_PROMPT, context, max_tokens=800, temperature=0.3)

    def _build_context(self, world: WorldState) -> str:
        """Build the mission context from current world state."""
        active = [d for d in world.drones if d.status == DroneStatus.ACTIVE]
        failed = [d.id for d in world.drones if d.status == DroneStatus.FAILED]
        low_battery = [
            d.id for d in world.drones if d.battery < 25 and d.status == DroneStatus.ACTIVE
        ]
        avg_battery = sum(d.battery for d in world.drones) / max(len(world.drones), 1)
        coverage = self._get_coverage(world.fog_grid)
        found = sum(1 for s in world.survivors if s.discovered)

        # Simple zone coverage approximation (quadrants)
        h, w = world.fog_grid.shape
        zone_coverage = {}
        for name, (r0, r1, c0, c1) in [
            ("NW", (0, h // 2, 0, w // 2)),
            ("NE", (0, h // 2, w // 2, w)),
            ("SW", (h // 2, h, 0, w // 2)),
            ("SE", (h // 2, h, w // 2, w)),
        ]:
            zone = world.fog_grid[r0:r1, c0:c1]
            total = zone.size
            explored = np.count_nonzero(zone != 0)
            zone_coverage[name] = (explored / total) * 100 if total > 0 else 0

        return build_mission_context(
            elapsed=world.elapsed,
            total_drones=len(world.drones),
            active_drones=len(active),
            failed_drones=failed,
            coverage_pct=coverage,
            survivors_found=found,
            total_survivors=len(world.survivors),
            fleet_avg_battery=avg_battery,
            low_battery_drones=low_battery,
            zone_coverage=zone_coverage,
            recent_events=self.event_log,
            human_overrides=self.human_overrides,
        )

    def _get_coverage(self, fog_grid: np.ndarray) -> float:
        total = fog_grid.size
        explored = np.count_nonzero(fog_grid != 0)
        return (explored / total) * 100
