"""Priority Market — unified auction-based task allocation.

Every priority source (PoC hotspots, operator zones, intel pins, LLM-injected
goals, recent survivor finds, future inputs) feeds a single `PriorityAsset`
stream. Drones bid on assets via one `bid()` function; a market clearing
step resolves assignments globally each tick.

This is the swarm's incorruptible baseline: adding a new input source is a
one-line append to the asset list and rebalances automatically. Re-tuning
behavior is a change to `PriorityWeights`, applied uniformly to all drones.

Design invariants the bid function enforces:
  1. A drone that can't round-trip an asset on its battery literally cannot
     bid for it (feasibility factor = 0).
  2. A drone deeply engaged (e.g. investigating a survivor) has a high
     switching cost — its bids for new assets are dampened, so it won't
     peel off.
  3. Distant drones bid proportionally less — far-side swarm members don't
     run across the map for a marginal target.
  4. Per-asset capacity caps prevent all drones stampeding one point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from src.simulation.types import Drone, DroneStatus, SimConfig, Vec3


@dataclass(frozen=True)
class PriorityAsset:
    """A single priority target. Position is in world-XZ meters."""

    asset_id: str
    x: float
    z: float
    radius: float        # effective area of influence (meters)
    value: float         # pre-scaled importance (caller applies source multiplier)
    source: str          # "poc_field" | "operator_high_zone" | "intel_pin" | ...
    expires_tick: int | None = None


@dataclass(frozen=True)
class PriorityWeights:
    """Tunable weights — the "constitution" shared across all drones.

    Edit this in one place; all drones re-evaluate with new values next tick.
    """

    # Per-source value scaling. An operator-marked asset should matter more
    # than a generic PoC hotspot; survivor finds more than either.
    source_value_scale: dict[str, float] = field(
        default_factory=lambda: {
            "poc_field": 1.0,
            "operator_high_zone": 5.0,
            "operator_low_zone": 0.3,
            "intel_pin": 3.0,
            "survivor_find": 8.0,
        }
    )

    # Switching cost per current task. Higher = stickier = harder to peel off.
    # Keyed by DroneTask.name strings (so this module doesn't import coordinator).
    switching_costs: dict[str, float] = field(
        default_factory=lambda: {
            "IDLE": 0.0,
            "FRONTIER_EXPLORING": 0.2,
            "MOVING_TO_ZONE": 0.5,
            "SWEEPING": 1.0,
            "INVESTIGATING": 4.0,
            "RETURNING_TO_BASE": 1e9,  # effectively cannot be peeled off RTB
        }
    )

    distance_penalty: float = 1.0          # coef on dist_to_asset in dist_factor
    base_return_penalty: float = 0.5       # coef on dist_to_base in dist_factor
    battery_safety_margin_pct: float = 10.0  # reserve battery % below which bids go 0
    # Drain multiplier to account for real per-tick drain vs nominal cruise rate
    # (movement + altitude + sensor combine to ~3.6× nominal). 4× keeps a small
    # safety buffer without over-restricting reach.
    drain_safety_multiplier: float = 4.0
    # Capacity per asset = ceil(effective_value / saturation). Using effective
    # value (post source-scale) lets operator-boosted zones absorb multiple
    # drones — real-world SAR clusters multiple drones on a hot area at
    # different altitudes / search patterns. Scaled to typical normalized PoC
    # peak values (~5e-4); a poc_field cell thus stays at capacity 1 while an
    # operator_high_zone cell (5× boost) absorbs ~5 drones.
    saturation_value_per_slot: float = 5e-4
    max_assignees_per_asset: int = 6         # hard cap per asset


def bid(
    drone: Drone,
    drone_task_name: str,
    asset: PriorityAsset,
    base_pos: Vec3,
    config: SimConfig,
    weights: PriorityWeights,
) -> float:
    """Compute a drone's bid for an asset.

    bid = value * source_scale
          / (1 + distance_penalty*dist_to_asset + base_return_penalty*dist_to_base)
          / (1 + switching_cost[current_task])

    Returns 0 if the drone can't feasibly reach the asset and return to base.
    """
    dx = drone.position.x - asset.x
    dz = drone.position.z - asset.z
    dist_to_asset = math.sqrt(dx * dx + dz * dz)

    bdx = asset.x - base_pos.x
    bdz = asset.z - base_pos.z
    dist_to_base = math.sqrt(bdx * bdx + bdz * bdz)

    # Battery feasibility: actual per-tick drain is ~3.6× nominal
    # (speed ×2.0 at cruise × altitude ×1.5 × sensor ×1.2). We use 4× as a
    # small safety buffer without stranding drones or over-restricting reach.
    speed_est = drone.max_speed * 0.85
    drain_est = config.drone_battery_drain_rate * weights.drain_safety_multiplier
    available = max(drone.battery - weights.battery_safety_margin_pct, 0.0)
    max_range = speed_est * (available / max(drain_est, 0.01))
    round_trip = dist_to_asset + dist_to_base
    if round_trip > max_range:
        return 0.0

    source_scale = weights.source_value_scale.get(asset.source, 1.0)
    effective_value = asset.value * source_scale

    dist_factor = (
        1.0
        + weights.distance_penalty * dist_to_asset
        + weights.base_return_penalty * dist_to_base
    )
    switch_cost = weights.switching_costs.get(drone_task_name, 0.0)
    return effective_value / (dist_factor * (1.0 + switch_cost))


def clear_market(
    drones: list[Drone],
    drone_tasks: dict[int, str],
    assets: list[PriorityAsset],
    base_pos: Vec3,
    config: SimConfig,
    weights: PriorityWeights,
    is_in_avoid_zone: Callable[[float, float], bool] | None = None,
) -> dict[int, PriorityAsset]:
    """Global one-shot auction — each drone gets at most one asset per tick.

    Per-asset capacity = max(1, min(max_assignees, ceil(value / saturation))).
    Drones are ranked by bid; top bidders claim capacity greedily.

    RTB-bound drones don't participate (their commitment is absolute).
    Assets inside avoid zones are filtered out up-front — they never receive
    any bid and drones already en route to them can be redirected by the
    caller (see _invalidate_targets_in_avoid_zones in coordinator).
    """
    if is_in_avoid_zone is not None:
        bidable = [a for a in assets if not is_in_avoid_zone(a.x, a.z)]
    else:
        bidable = list(assets)
    if not bidable:
        return {}

    # Capacity uses effective value (value × source_scale), so an operator
    # high zone with source_scale 5× soaks up ~5 drones per cell instead of 1.
    capacity: dict[str, int] = {}
    for a in bidable:
        effective = a.value * weights.source_value_scale.get(a.source, 1.0)
        capacity[a.asset_id] = max(
            1,
            min(
                weights.max_assignees_per_asset,
                math.ceil(effective / max(weights.saturation_value_per_slot, 1e-6)),
            ),
        )

    bid_tuples: list[tuple[float, int, PriorityAsset]] = []
    for drone in drones:
        if drone.status != DroneStatus.ACTIVE:
            continue
        task = drone_tasks.get(drone.id, "IDLE")
        if task == "RETURNING_TO_BASE":
            continue
        for asset in bidable:
            score = bid(drone, task, asset, base_pos, config, weights)
            if score > 0:
                bid_tuples.append((score, drone.id, asset))

    # Sort descending: highest bids claim capacity first. Tiebreak on drone_id
    # for determinism.
    bid_tuples.sort(key=lambda t: (-t[0], t[1]))

    assignments: dict[int, PriorityAsset] = {}
    for _score, drone_id, asset in bid_tuples:
        if drone_id in assignments:
            continue
        if capacity.get(asset.asset_id, 0) <= 0:
            continue
        assignments[drone_id] = asset
        capacity[asset.asset_id] -= 1

    return assignments
