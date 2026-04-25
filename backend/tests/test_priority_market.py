"""Tests for the priority market (auction-based task allocation)."""

from __future__ import annotations

from src.agents.priority_market import (
    PriorityAsset,
    PriorityWeights,
    bid,
    clear_market,
)
from src.simulation.types import Drone, DroneStatus, SimConfig, Vec3


def _drone(
    drone_id: int,
    x: float = 0.0,
    z: float = 0.0,
    battery: float = 100.0,
    status: DroneStatus = DroneStatus.ACTIVE,
) -> Drone:
    return Drone(id=drone_id, position=Vec3(x, 50.0, z), battery=battery, status=status)


def _cfg() -> SimConfig:
    return SimConfig()


def _asset(
    asset_id: str = "a1",
    x: float = 500.0,
    z: float = 0.0,
    value: float = 5e-4,  # realistic normalized-PoC peak
    source: str = "poc_field",
) -> PriorityAsset:
    return PriorityAsset(
        asset_id=asset_id,
        x=x,
        z=z,
        radius=50.0,
        value=value,
        source=source,
    )


# ---------------------------------------------------------------- bid() tests


def test_bid_infeasible_battery_returns_zero() -> None:
    """A drone with no battery reserve cannot bid on a distant asset."""
    d = _drone(0, x=0.0, z=0.0, battery=15.0)  # 5% usable after margin
    a = _asset(x=5000.0, z=0.0)
    score = bid(d, "IDLE", a, Vec3(0, 0, 0), _cfg(), PriorityWeights())
    assert score == 0.0


def test_bid_closer_drone_wins() -> None:
    """Bid scales inversely with distance — closer drone wins."""
    near = _drone(0, x=400.0, z=0.0)
    far = _drone(1, x=2000.0, z=0.0)
    a = _asset(x=500.0, z=0.0, value=1.0)
    w = PriorityWeights()
    b_near = bid(near, "IDLE", a, Vec3(0, 0, 0), _cfg(), w)
    b_far = bid(far, "IDLE", a, Vec3(0, 0, 0), _cfg(), w)
    assert b_near > b_far > 0


def test_bid_switching_cost_dampens_investigating() -> None:
    """An INVESTIGATING drone bids less for new assets than an IDLE drone."""
    d = _drone(0, x=400.0, z=0.0)
    a = _asset(x=500.0, z=0.0)
    w = PriorityWeights()
    b_idle = bid(d, "IDLE", a, Vec3(0, 0, 0), _cfg(), w)
    b_inv = bid(d, "INVESTIGATING", a, Vec3(0, 0, 0), _cfg(), w)
    # INVESTIGATING has switching_cost 4.0, IDLE has 0.0 → bid divided by 5× vs 1×
    assert b_idle > b_inv > 0
    assert b_idle / b_inv > 4.0


def test_bid_source_scale_prefers_operator_zone() -> None:
    """operator_high_zone source boosts value over plain poc_field."""
    d = _drone(0, x=0.0, z=0.0)
    a_zone = _asset(asset_id="z", x=500.0, z=0.0, value=1.0, source="operator_high_zone")
    a_poc = _asset(asset_id="p", x=500.0, z=0.0, value=1.0, source="poc_field")
    w = PriorityWeights()
    assert bid(d, "IDLE", a_zone, Vec3(0, 0, 0), _cfg(), w) > bid(
        d,
        "IDLE",
        a_poc,
        Vec3(0, 0, 0),
        _cfg(),
        w,
    )


def test_bid_base_return_penalty_punishes_far_base() -> None:
    """Two equal-distance assets, one farther from base → nearer-base wins."""
    d = _drone(0, x=0.0, z=0.0)
    base = Vec3(0, 0, 0)
    a_near_base = _asset(asset_id="n", x=500.0, z=0.0)
    a_far_base = _asset(asset_id="f", x=0.0, z=5000.0)  # farther round-trip base
    w = PriorityWeights()
    b_near = bid(d, "IDLE", a_near_base, base, _cfg(), w)
    # Place drone at (0, 0, 2500) so both assets are ~2500m from drone
    d2 = _drone(0, x=0.0, z=2500.0)
    # Now near-base asset is sqrt(500^2 + 2500^2) ≈ 2549m from drone
    # far-base asset is 2500m from drone, 5000m from base
    b_far = bid(d2, "IDLE", a_far_base, base, _cfg(), w)
    b_near2 = bid(d2, "IDLE", a_near_base, base, _cfg(), w)
    assert b_near2 > b_far
    _ = b_near  # silence unused


# ---------------------------------------------------------------- clear_market()


def test_clear_market_assigns_each_drone_at_most_once() -> None:
    d0 = _drone(0, x=100.0, z=0.0)
    d1 = _drone(1, x=120.0, z=0.0)
    # Default asset value 5e-4 → capacity 1 per asset at default saturation.
    a0 = _asset(asset_id="a0", x=500.0, z=0.0)
    a1 = _asset(asset_id="a1", x=-500.0, z=0.0)
    assignments = clear_market(
        drones=[d0, d1],
        drone_tasks={0: "IDLE", 1: "IDLE"},
        assets=[a0, a1],
        base_pos=Vec3(0, 0, 0),
        config=_cfg(),
        weights=PriorityWeights(),
    )
    assert set(assignments.keys()) == {0, 1}
    # Each poc_field asset has capacity 1 at default saturation → drones
    # must split.
    assert assignments[0].asset_id != assignments[1].asset_id


def test_clear_market_respects_per_asset_capacity() -> None:
    """A poc_field asset caps at capacity 1 — only one drone can take it."""
    d0 = _drone(0, x=100.0, z=0.0)
    d1 = _drone(1, x=110.0, z=0.0)
    a_big = _asset(asset_id="big", x=500.0, z=0.0)  # poc_field → cap 1
    a_far = _asset(asset_id="far", x=-3000.0, z=0.0)  # poc_field → cap 1
    assignments = clear_market(
        drones=[d0, d1],
        drone_tasks={0: "IDLE", 1: "IDLE"},
        assets=[a_big, a_far],
        base_pos=Vec3(0, 0, 0),
        config=_cfg(),
        weights=PriorityWeights(),
    )
    big_count = sum(1 for a in assignments.values() if a.asset_id == "big")
    assert big_count == 1


def test_clear_market_operator_zone_absorbs_multiple_drones() -> None:
    """An operator_high_zone asset (5× source scale) should take ~5 drones."""
    # Many drones clustered near the zone asset
    drones = [_drone(i, x=100.0 + i * 5, z=0.0) for i in range(8)]
    tasks = {d.id: "IDLE" for d in drones}
    zone_asset = _asset(
        asset_id="zone_hot",
        x=400.0,
        z=0.0,
        value=5e-4,
        source="operator_high_zone",
    )
    # Include a distant fallback so drones that don't win the zone have
    # somewhere to go.
    fallback = _asset(asset_id="fallback", x=-2000.0, z=0.0, value=5e-4)
    assignments = clear_market(
        drones=drones,
        drone_tasks=tasks,
        assets=[zone_asset, fallback],
        base_pos=Vec3(0, 0, 0),
        config=_cfg(),
        weights=PriorityWeights(),
    )
    in_zone = sum(1 for a in assignments.values() if a.asset_id == "zone_hot")
    # Effective value = 5e-4 × 5.0 = 2.5e-3; at saturation 5e-4 → capacity 5.
    assert in_zone == 5


def test_clear_market_excludes_rtb_drones() -> None:
    d0 = _drone(0, x=100.0, z=0.0)
    d1 = _drone(1, x=120.0, z=0.0)
    a = _asset()
    assignments = clear_market(
        drones=[d0, d1],
        drone_tasks={0: "IDLE", 1: "RETURNING_TO_BASE"},
        assets=[a],
        base_pos=Vec3(0, 0, 0),
        config=_cfg(),
        weights=PriorityWeights(),
    )
    assert 0 in assignments
    assert 1 not in assignments


def test_clear_market_filters_avoid_zones() -> None:
    """Assets inside an avoid zone get no bids."""
    d = _drone(0, x=0.0, z=0.0)
    a_ok = _asset(asset_id="ok", x=500.0, z=0.0)
    a_avoid = _asset(asset_id="bad", x=-500.0, z=0.0)

    def is_avoid(x: float, z: float) -> bool:
        return x < 0.0  # whole western half is avoid

    assignments = clear_market(
        drones=[d],
        drone_tasks={0: "IDLE"},
        assets=[a_ok, a_avoid],
        base_pos=Vec3(0, 0, 0),
        config=_cfg(),
        weights=PriorityWeights(),
        is_in_avoid_zone=is_avoid,
    )
    assert 0 in assignments
    assert assignments[0].asset_id == "ok"


def test_bid_switching_cost_is_multiplicative_and_uniform() -> None:
    """Switching cost scales every asset's bid by the same factor, so it
    can't flip the relative ranking — it only dampens magnitudes. That
    means a drone doing high-cost work (e.g. INVESTIGATING) has smaller
    bids overall and will lose auctions to IDLE drones, but its internal
    ranking of assets is preserved.
    """
    d = _drone(0, x=100.0, z=0.0)
    # Both well within battery round-trip (max range ~4590m at 100% battery).
    near = _asset(asset_id="n", x=200.0, z=0.0, value=1.0, source="poc_field")
    far = _asset(asset_id="f", x=1200.0, z=0.0, value=1.0, source="operator_high_zone")
    w = PriorityWeights()
    b_idle_near = bid(d, "IDLE", near, Vec3(0, 0, 0), _cfg(), w)
    b_inv_near = bid(d, "INVESTIGATING", near, Vec3(0, 0, 0), _cfg(), w)
    b_idle_far = bid(d, "IDLE", far, Vec3(0, 0, 0), _cfg(), w)
    b_inv_far = bid(d, "INVESTIGATING", far, Vec3(0, 0, 0), _cfg(), w)
    # Switching cost dampens both bids by the same multiplicative factor.
    assert abs((b_idle_near / b_inv_near) - (b_idle_far / b_inv_far)) < 1e-6
    # Within a drone, rankings are preserved across switching cost.
    assert (b_idle_near > b_idle_far) == (b_inv_near > b_inv_far)
    # INVESTIGATING bids are smaller overall → an IDLE drone will outbid
    # the INVESTIGATING drone for the same asset (this is what prevents
    # productive drones from being peeled off).
    assert b_idle_near > b_inv_near
    assert b_idle_far > b_inv_far
