"""Tests for the Phase 5D adaptive weights layer."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.adaptive_weights import (
    ADAPT_MAX,
    ADAPT_MIN,
    FIND_CREDIT,
    OVERRIDE_CREDIT,
    TRUST_MAX,
    TRUST_MIN,
    UNUSED_DEBIT,
    AdaptiveWeights,
)
from src.agents.priority_market import PriorityWeights

# --- Test fixtures (mimic PriorityZone / IntelPin shapes) --------------------


@dataclass
class _FakeZone:
    zone_id: str
    priority: str
    xmin: float = 0.0
    zmin: float = 0.0
    xmax: float = 100.0
    zmax: float = 100.0

    def contains(self, x: float, z: float) -> bool:
        return self.xmin <= x <= self.xmax and self.zmin <= z <= self.zmax


@dataclass
class _FakePin:
    pin_id: str
    x: float
    z: float
    radius: float


# --- Trust slider ------------------------------------------------------------


def test_trust_set_within_bounds() -> None:
    a = AdaptiveWeights()
    a.set_trust(2.5)
    assert a.operator_trust == 2.5


def test_trust_clamps_to_bounds() -> None:
    a = AdaptiveWeights()
    a.set_trust(-1.0)
    assert a.operator_trust == TRUST_MIN
    a.set_trust(999.0)
    assert a.operator_trust == TRUST_MAX


def test_trust_only_scales_operator_sources() -> None:
    a = AdaptiveWeights()
    a.set_trust(2.0)
    eff = a.effective_weights()
    base = PriorityWeights()
    # Operator sources doubled
    assert eff.source_value_scale["intel_pin"] == base.source_value_scale["intel_pin"] * 2.0
    assert (
        eff.source_value_scale["operator_high_zone"]
        == base.source_value_scale["operator_high_zone"] * 2.0
    )
    # PoC + survivor_find unaffected (system-truth sources)
    assert eff.source_value_scale["poc_field"] == base.source_value_scale["poc_field"]
    assert eff.source_value_scale["survivor_find"] == base.source_value_scale["survivor_find"]


# --- Find attribution --------------------------------------------------------


def test_find_inside_high_zone_credits_operator_high_zone() -> None:
    a = AdaptiveWeights()
    zone = _FakeZone("z1", "high", 0, 0, 100, 100)
    credited = a.record_survivor_find(
        tick=10,
        survivor_xz=(50.0, 50.0),
        active_zones=[zone],
        active_pins=[],
    )
    assert credited == [("operator_high_zone", "zone_z1")]
    # Scale nudged up
    scale = a.learned_source_scale["operator_high_zone"]
    assert abs(scale - (1.0 + FIND_CREDIT)) < 1e-9


def test_find_outside_zone_credits_nothing() -> None:
    a = AdaptiveWeights()
    zone = _FakeZone("z1", "high", 0, 0, 100, 100)
    credited = a.record_survivor_find(
        tick=10,
        survivor_xz=(500.0, 500.0),
        active_zones=[zone],
        active_pins=[],
    )
    assert credited == []
    assert a.learned_source_scale == {}


def test_find_inside_avoid_zone_does_not_credit() -> None:
    a = AdaptiveWeights()
    zone = _FakeZone("z_avoid", "avoid", 0, 0, 100, 100)
    credited = a.record_survivor_find(
        tick=10,
        survivor_xz=(50.0, 50.0),
        active_zones=[zone],
        active_pins=[],
    )
    assert credited == []


def test_find_inside_intel_pin_radius_credits_pin() -> None:
    a = AdaptiveWeights()
    pin = _FakePin(pin_id="p1", x=100.0, z=100.0, radius=50.0)
    credited = a.record_survivor_find(
        tick=10,
        survivor_xz=(120.0, 120.0),  # ~28m from pin, well inside 50m
        active_zones=[],
        active_pins=[pin],
    )
    assert credited == [("intel_pin", "intel_p1")]


def test_find_outside_intel_pin_radius_does_not_credit() -> None:
    a = AdaptiveWeights()
    pin = _FakePin(pin_id="p1", x=100.0, z=100.0, radius=20.0)
    credited = a.record_survivor_find(
        tick=10,
        survivor_xz=(200.0, 200.0),  # ~141m from pin, well outside
        active_zones=[],
        active_pins=[pin],
    )
    assert credited == []


# --- Unused / override / decay -----------------------------------------------


def test_unused_debits_source() -> None:
    a = AdaptiveWeights()
    a.record_asset_unused(tick=5, source="intel_pin", asset_id="p1")
    assert a.learned_source_scale["intel_pin"] == 1.0 - UNUSED_DEBIT


def test_unused_ignores_non_operator_source() -> None:
    a = AdaptiveWeights()
    a.record_asset_unused(tick=5, source="poc_field", asset_id="x")
    assert "poc_field" not in a.learned_source_scale


def test_override_raises_switching_cost_for_task() -> None:
    a = AdaptiveWeights()
    a.record_operator_override(tick=5, overridden_task="SWEEPING")
    assert a.learned_switching_cost["SWEEPING"] == 1.0 + OVERRIDE_CREDIT


def test_bounds_cap_at_adapt_max_and_adapt_min() -> None:
    a = AdaptiveWeights()
    for _ in range(100):
        a.record_survivor_find(
            tick=1,
            survivor_xz=(50, 50),
            active_zones=[_FakeZone("z1", "high", 0, 0, 100, 100)],
            active_pins=[],
        )
    assert a.learned_source_scale["operator_high_zone"] == ADAPT_MAX

    a2 = AdaptiveWeights()
    for _ in range(200):
        a2.record_asset_unused(tick=1, source="intel_pin", asset_id="p")
    assert a2.learned_source_scale["intel_pin"] == ADAPT_MIN


def test_tick_decay_pulls_toward_neutral() -> None:
    a = AdaptiveWeights()
    a.learned_source_scale["intel_pin"] = 1.5
    a.learned_source_scale["operator_low_zone"] = 0.5
    for _ in range(500):  # 500 ticks × 0.0001 = 0.05 of decay
        a.tick_decay()
    assert 1.44 <= a.learned_source_scale["intel_pin"] <= 1.46
    assert 0.54 <= a.learned_source_scale["operator_low_zone"] <= 0.56


def test_effective_weights_combines_trust_and_learning() -> None:
    a = AdaptiveWeights()
    # Learn intel_pin is great
    for _ in range(10):
        a.record_survivor_find(
            tick=1,
            survivor_xz=(50, 50),
            active_zones=[],
            active_pins=[_FakePin("p", 50, 50, 100)],
        )
    # Then bump trust further
    a.set_trust(1.5)
    eff = a.effective_weights()
    base = PriorityWeights()
    # Effective = base × learned × trust
    expected = base.source_value_scale["intel_pin"] * a.learned_source_scale["intel_pin"] * 1.5
    assert abs(eff.source_value_scale["intel_pin"] - expected) < 1e-6


def test_summary_reports_state() -> None:
    a = AdaptiveWeights()
    a.set_trust(1.3)
    a.record_operator_override(tick=1, overridden_task="SWEEPING")
    s = a.summary()
    assert s["operator_trust"] == 1.3
    assert "SWEEPING" in s["learned_switching_cost"]
    assert s["recent_outcome_count"] == 1
