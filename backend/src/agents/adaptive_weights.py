"""Adaptive priority weights — the swarm learns which operator signals pay off.

Phase 5D. Sits on top of the priority market (Phase 5B/5C) and nudges
`PriorityWeights.source_value_scale` over time based on observed outcomes:

  * Survivor found inside an operator-placed asset → +credit to that source.
  * Operator asset expires (TTL ran out / operator deleted) with no finds
    attributed to it → -credit.
  * Operator overrides a drone mid-task → +credit to that current task's
    switching cost (drones were too easily stuck, raise the cost).

Plus a `operator_trust` scalar (the Phase 5D "trust/autonomy slider" from
general_plan.md) that multiplies on top — 1.0 = defaults, >1 = operator
hints matter more, <1 = system defers to its own signals.

Design notes:
- All adjustments are BOUNDED and DECAY toward neutral over time, so a
  noisy run can't lock the swarm into a bad weight forever.
- The bandit logic is deliberately simple (proportional nudges, not
  Thompson sampling) so it stays interpretable and testable. It matches
  the feedback pattern "learn from the operator without overcorrecting"
  which is the real-world SAR posture.
- This module is import-cheap: no numpy, no LLM calls, no async. It's a
  tiny state machine coordinator owns and ticks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from src.agents.priority_market import PriorityWeights

# Bounds on the learned adjustment multiplier (how far learning can shift
# a source's scale). Half to triple the base is plenty for interpretability.
ADAPT_MIN = 0.5
ADAPT_MAX = 3.0

# Per-event nudges. Small so the system adapts gradually.
FIND_CREDIT = 0.08          # +8% on a survivor/evidence find attributable to this source
UNUSED_DEBIT = 0.02         # -2% when an operator asset expires with no finds
OVERRIDE_CREDIT = 0.04      # +4% to switching_cost on operator override of a drone task

# Gentle decay toward 1.0 each tick — so learning doesn't accumulate forever.
# At 10Hz, this halves the deviation from 1.0 every ~700s (~12 minutes).
DECAY_PER_TICK = 0.0001

# Trust slider bounds — keep the operator in a sane range.
TRUST_MIN = 0.0    # "operator hints are purely advisory"
TRUST_MAX = 3.0    # "operator is almost always right"

# Sources that operator_trust scales (only operator-originated sources — we
# never scale poc_field or survivor_find by trust, those are system-truth).
OPERATOR_SOURCES = ("operator_high_zone", "operator_low_zone", "intel_pin")


@dataclass(frozen=True)
class OutcomeRecord:
    """One attributable event in the outcome ring buffer."""

    tick: int
    source: str                 # PriorityAsset.source, or a DroneTask name for overrides
    kind: str                   # "find" | "unused" | "override"
    asset_id: str = ""          # for traceability


@dataclass
class AdaptiveWeights:
    """Mutable adaptive layer on top of an immutable base `PriorityWeights`.

    Coordinator owns exactly one of these, calls `record_*` on events, and
    reads `effective_weights()` before building the market each tick.
    """

    base: PriorityWeights = field(default_factory=PriorityWeights)
    operator_trust: float = 1.0
    # Multiplicative adjustments learned from outcomes (start at 1.0).
    learned_source_scale: dict[str, float] = field(default_factory=dict)
    learned_switching_cost: dict[str, float] = field(default_factory=dict)
    # Ring-buffered outcome log for introspection / testing / UI.
    outcomes: list[OutcomeRecord] = field(default_factory=list)
    max_outcomes: int = 500

    # --------------------------------------------------------------- api

    def set_trust(self, value: float) -> None:
        self.operator_trust = max(TRUST_MIN, min(TRUST_MAX, float(value)))

    def record_survivor_find(
        self,
        tick: int,
        survivor_xz: tuple[float, float],
        active_zones: Iterable,       # PriorityZone objects (must expose .contains(x,z) + .priority + .zone_id)
        active_pins: Iterable,        # IntelPin objects (must expose .x, .z, .radius, .pin_id)
    ) -> list[tuple[str, str]]:
        """Attribute a survivor find to any operator assets overlapping it.

        Returns (source, asset_key) tuples for each asset credited. The
        coordinator stores these in `_credited_assets` so we don't later
        debit the same asset for being "unused" when it has actually
        contributed.
        """
        credited: list[tuple[str, str]] = []
        sx, sz = survivor_xz
        for zone in active_zones:
            try:
                inside = zone.contains(sx, sz)
            except Exception:
                inside = False
            if not inside:
                continue
            priority = getattr(zone, "priority", "")
            if priority == "high":
                source = "operator_high_zone"
            elif priority == "low":
                source = "operator_low_zone"
            else:
                continue  # avoid zones don't earn credit for finds inside them
            zone_id = getattr(zone, "zone_id", "")
            self._bump_source(source, +FIND_CREDIT)
            self._log(tick, source, "find", zone_id)
            credited.append((source, f"zone_{zone_id}"))
        for pin in active_pins:
            dx = sx - pin.x
            dz = sz - pin.z
            if math.hypot(dx, dz) <= pin.radius:
                pin_id = getattr(pin, "pin_id", "")
                self._bump_source("intel_pin", +FIND_CREDIT)
                self._log(tick, "intel_pin", "find", pin_id)
                credited.append(("intel_pin", f"intel_{pin_id}"))
        return credited

    def record_asset_unused(self, tick: int, source: str, asset_id: str) -> None:
        """Called when an operator zone/pin disappears without finds credited."""
        if source not in OPERATOR_SOURCES:
            return
        self._bump_source(source, -UNUSED_DEBIT)
        self._log(tick, source, "unused", asset_id)

    def record_operator_override(self, tick: int, overridden_task: str) -> None:
        """Called when the operator moves/rtbs a drone that was mid-task —
        a signal that its current task was the wrong thing to be doing."""
        if not overridden_task:
            return
        self._bump_switching(overridden_task, +OVERRIDE_CREDIT)
        self._log(tick, overridden_task, "override", "")

    def tick_decay(self) -> None:
        """Pull learned adjustments gently toward 1.0 each tick."""
        for tbl in (self.learned_source_scale, self.learned_switching_cost):
            for k in list(tbl.keys()):
                v = tbl[k]
                if v > 1.0:
                    tbl[k] = max(1.0, v - DECAY_PER_TICK)
                elif v < 1.0:
                    tbl[k] = min(1.0, v + DECAY_PER_TICK)

    # ------------------------------------------------------- derived weights

    def effective_weights(self) -> PriorityWeights:
        """Return a fresh PriorityWeights with trust + learned adjustments baked in."""
        # source_value_scale: base × learned × (trust for operator sources)
        scales: dict[str, float] = dict(self.base.source_value_scale)
        for src, base in self.base.source_value_scale.items():
            learned = self.learned_source_scale.get(src, 1.0)
            trust = self.operator_trust if src in OPERATOR_SOURCES else 1.0
            scales[src] = base * learned * trust

        # switching_costs: base × learned
        switching: dict[str, float] = dict(self.base.switching_costs)
        for task, base in self.base.switching_costs.items():
            learned = self.learned_switching_cost.get(task, 1.0)
            switching[task] = base * learned

        return PriorityWeights(
            source_value_scale=scales,
            switching_costs=switching,
            distance_penalty=self.base.distance_penalty,
            base_return_penalty=self.base.base_return_penalty,
            battery_safety_margin_pct=self.base.battery_safety_margin_pct,
            drain_safety_multiplier=self.base.drain_safety_multiplier,
            saturation_value_per_slot=self.base.saturation_value_per_slot,
            max_assignees_per_asset=self.base.max_assignees_per_asset,
        )

    def summary(self) -> dict:
        """Compact snapshot for HUD / chat status / tests."""
        return {
            "operator_trust": round(self.operator_trust, 3),
            "learned_source_scale": {
                k: round(v, 3) for k, v in self.learned_source_scale.items()
            },
            "learned_switching_cost": {
                k: round(v, 3) for k, v in self.learned_switching_cost.items()
            },
            "recent_outcome_count": len(self.outcomes),
            "recent_outcome_mix": self._outcome_mix(),
        }

    # ------------------------------------------------------- internals

    def _bump_source(self, source: str, delta: float) -> None:
        cur = self.learned_source_scale.get(source, 1.0)
        new = max(ADAPT_MIN, min(ADAPT_MAX, cur + delta))
        self.learned_source_scale[source] = new

    def _bump_switching(self, task: str, delta: float) -> None:
        cur = self.learned_switching_cost.get(task, 1.0)
        new = max(ADAPT_MIN, min(ADAPT_MAX, cur + delta))
        self.learned_switching_cost[task] = new

    def _log(self, tick: int, source: str, kind: str, asset_id: str) -> None:
        self.outcomes.append(OutcomeRecord(tick=tick, source=source, kind=kind, asset_id=asset_id))
        if len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes :]

    def _outcome_mix(self) -> dict[str, int]:
        mix: dict[str, int] = {}
        for rec in self.outcomes[-100:]:  # recent slice
            key = f"{rec.kind}:{rec.source}"
            mix[key] = mix.get(key, 0) + 1
        return mix
