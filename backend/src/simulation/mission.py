"""Mission templates — prior intelligence and scenario configuration.

A `SearchMission` packages everything the simulator needs to model a specific
SAR scenario:

  * Where survivors are *actually* placed (ground truth) — `generate_clusters`
  * Where the search planner *thinks* they are (Bayesian prior) — `seed_poc_grid`
  * Where the base of operations sits, tuned so the prior is reachable round-trip
  * A short operator briefing with the known facts

The split between ground truth (clusters) and prior (PoC seeding) is deliberate.
Phase 3 will widen this gap — the prior will be wrong in informative ways, and
finding evidence will narrow it. For Phase 2 the prior is shaped like the
ground truth but smoothed: the planner has roughly correct intel, not omniscient
intel.

Base placement is the architectural workaround for the reach bottleneck (see
`memory/project_reach_bottleneck.md`). Until Phase 8 ships forward operating
bases, each scenario picks a base location near the high-mass region of its
prior so a fresh drone can make the round trip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.simulation.search_map import SearchMap
from src.simulation.types import Evidence, EvidenceKind, Vec3

# Survivor clusters use the same 4-tuple format the ChunkedWorld already
# expects: (center_x, center_z, radius_meters, weight). Weight is a fraction
# of the configured survivor_count that this cluster receives.
Cluster = tuple[float, float, float, float]


class SeedablePrior(Protocol):
    """Anything that can seed a SearchMap. Each mission implements this."""

    def seed_poc_grid(self, grid: SearchMap) -> None: ...


@dataclass
class SearchMission:
    """Base class for SAR mission scenarios.

    Subclasses override `generate_clusters` and `seed_poc_grid` with
    domain-specific logic. They also pick a `base_position` tuned so the
    prior's high-mass region is within fresh-drone round-trip range.

    Attributes:
        name: Short identifier, used in UI selectors and logs.
        title: Human-readable title for the briefing overlay.
        description: 2-3 sentence scenario summary for the operator.
        known_facts: Bullet-style facts pinned to the briefing/minimap.
        base_position: Where the swarm launches from.
        survivor_count_range: (min, max) survivors actually present.
        survival_window_seconds: Hard time before survivors can no longer be saved.
        clusters: Ground-truth survivor placement (set by `generate_clusters`).
        world_size: World extent in meters (square world assumed).
        seed: RNG seed used by the mission generator.
    """

    name: str
    title: str
    description: str
    known_facts: list[str]
    base_position: Vec3
    survivor_count_range: tuple[int, int]
    survival_window_seconds: float
    clusters: list[Cluster]
    world_size: int
    seed: int
    # Anchor positions for the briefing overlay (last-known, trailhead, etc.).
    # Renderable as map pins on the frontend.
    intel_pins: list[dict] = field(default_factory=list)
    # Planted clues (footprints, debris, signal fires) drones can discover.
    # Each evidence triggers a Bayesian posterior update when found. Empty
    # for missions that don't model a survivor trail.
    evidence: list[Evidence] = field(default_factory=list)

    def seed_poc_grid(self, grid: SearchMap) -> None:
        """Seed the PoC prior on the search map.

        Default implementation lays a Gaussian over each ground-truth cluster
        — Phase 3 will replace this with prior-only sources (last-known
        position, drift integration) that don't see the cluster centers.
        """
        for cx, cz, cr, cw in self.clusters:
            grid.add_gaussian(
                center_world=(cx, cz),
                radius_meters=max(cr, 200.0),
                weight=float(cw),
            )
        # Weak uniform baseline so off-prior areas aren't "impossible"
        grid.poc += np.float32(0.001)
        grid.normalize(target_mass=1.0)

    def to_briefing_dict(self) -> dict:
        """Serialize for the `mission_briefing` WebSocket message."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "known_facts": list(self.known_facts),
            "base_position": [self.base_position.x, self.base_position.y, self.base_position.z],
            "survival_window_seconds": self.survival_window_seconds,
            "intel_pins": list(self.intel_pins),
        }


# ---------------------------------------------------------------------------
# Geometry helpers — used by multiple missions
# ---------------------------------------------------------------------------


def _clamp_to_world(
    x: float, z: float, world_size: int, margin: float = 50.0
) -> tuple[float, float]:
    lo = margin
    hi = world_size - margin
    return max(lo, min(hi, x)), max(lo, min(hi, z))


def _cone_clusters(
    apex_x: float,
    apex_z: float,
    bearing_rad: float,
    length_m: float,
    half_angle_rad: float,
    n_segments: int,
    weights: list[float],
    spread_at_apex: float,
    spread_at_far: float,
    world_size: int,
) -> list[Cluster]:
    """Lay clusters along a cone — used for crash debris and search corridors.

    Apex is the known starting point (last radar ping, trailhead). The cone
    fans out along `bearing_rad` for `length_m` meters. Each segment gets one
    cluster centered on the cone axis with a radius that grows linearly from
    `spread_at_apex` (uncertainty at the apex) to `spread_at_far` (uncertainty
    at the far end).
    """
    if len(weights) != n_segments:
        msg = f"weights length {len(weights)} != n_segments {n_segments}"
        raise ValueError(msg)
    out: list[Cluster] = []
    for i in range(n_segments):
        # Place each segment at the midpoint of its slice along the axis.
        t = (i + 0.5) / n_segments
        dist = t * length_m
        cx = apex_x + math.cos(bearing_rad) * dist
        cz = apex_z + math.sin(bearing_rad) * dist
        cx, cz = _clamp_to_world(cx, cz, world_size)
        radius = spread_at_apex + (spread_at_far - spread_at_apex) * t
        # Half-angle widens the cone — incorporate it into the radius.
        radius = max(radius, dist * math.tan(half_angle_rad))
        out.append((cx, cz, radius, weights[i]))
    return out


# ---------------------------------------------------------------------------
# Evidence trail generation
# ---------------------------------------------------------------------------


def _plant_trail_evidence(
    start_x: float,
    start_z: float,
    end_x: float,
    end_z: float,
    world_size: int,
    rng: np.random.Generator,
    *,
    signal_fire_at_end: bool = True,
) -> list[Evidence]:
    """Plant a sparse line of evidence from `start` toward `end`.

    The trail models the survivor's actual path between two mission
    anchors — e.g., last-known position → impact site for an aircraft
    crash, or trailhead → current hiker position for a lost hiker. Drones
    that happen to pass near these clues discover them and the PoC
    posterior narrows in real time.

    Returns a small mix of:
      * 2–3 footprints spaced along the trail (directional, heading
        toward `end`)
      * 1 debris cluster near the midpoint (isotropic ring)
      * 1 signal_fire at the end (high confidence, near the survivors)

    Kept deliberately sparse so discovery remains a visible event, not
    ambient noise. All positions use y=0; the planter doesn't know terrain
    height and the frontend samples elevation at render time.
    """
    evidence: list[Evidence] = []
    dx = end_x - start_x
    dz = end_z - start_z
    trail_len = math.hypot(dx, dz)
    if trail_len < 1.0:
        return evidence

    heading = math.atan2(dx, dz)  # heading 0 = +Z/north, atan2(east, north)

    # Footprints along the trail at 35% / 60% / 80% of its length. Each
    # point jittered perpendicular to the trail so they don't look like a
    # ruler-straight line.
    perp_x = -dz / trail_len
    perp_z = dx / trail_len
    next_id = 0
    for frac, age in [(0.35, 1.5), (0.60, 1.0), (0.80, 0.5)]:
        base_x = start_x + dx * frac
        base_z = start_z + dz * frac
        jitter = float(rng.uniform(-60.0, 60.0))
        fx = base_x + perp_x * jitter
        fz = base_z + perp_z * jitter
        fx, fz = _clamp_to_world(fx, fz, world_size, margin=30.0)
        evidence.append(
            Evidence(
                id=next_id,
                position=Vec3(fx, 0.0, fz),
                kind=EvidenceKind.FOOTPRINT.value,
                confidence=0.6,
                heading=heading,
                age_hours=age,
            )
        )
        next_id += 1

    # Debris cluster near the midpoint — isotropic, slightly off-axis so
    # it's a distinct find rather than colocated with a footprint.
    mid_frac = 0.50
    mid_x = start_x + dx * mid_frac
    mid_z = start_z + dz * mid_frac
    mid_jitter = float(rng.uniform(80.0, 180.0))
    mid_sign = 1.0 if rng.random() < 0.5 else -1.0
    mx = mid_x + perp_x * mid_jitter * mid_sign
    mz = mid_z + perp_z * mid_jitter * mid_sign
    mx, mz = _clamp_to_world(mx, mz, world_size, margin=30.0)
    evidence.append(
        Evidence(
            id=next_id,
            position=Vec3(mx, 0.0, mz),
            kind=EvidenceKind.DEBRIS.value,
            confidence=0.45,
            heading=None,
            age_hours=2.0,
        )
    )
    next_id += 1

    # Signal fire near the endpoint — high confidence survivor indicator.
    if signal_fire_at_end:
        sf_x = end_x + float(rng.uniform(-120.0, 120.0))
        sf_z = end_z + float(rng.uniform(-120.0, 120.0))
        sf_x, sf_z = _clamp_to_world(sf_x, sf_z, world_size, margin=30.0)
        evidence.append(
            Evidence(
                id=next_id,
                position=Vec3(sf_x, 0.0, sf_z),
                kind=EvidenceKind.SIGNAL_FIRE.value,
                confidence=0.9,
                heading=None,
                age_hours=0.25,
            )
        )

    return evidence


def _plant_drift_evidence(
    lkp_x: float,
    lkp_z: float,
    com_x: float,
    com_z: float,
    leeway_uncertainty: float,
    world_size: int,
    rng: np.random.Generator,
) -> list[Evidence]:
    """Maritime drift trail — debris carried downstream from LKP toward drift COM.

    Geometry follows surface current + leeway: wreckage/life-jacket debris
    strewn along the drift line, weathered near the LKP (oldest) and fresher
    near the leading edge. A directional "slick" marker (repurposed footprint
    kind with heading=current bearing) sits mid-drift. A signal flare/dye
    marker at the projected COM represents the survivor's current position.

    Perpendicular jitter is bounded by the leeway uncertainty — the same
    parameter that sets the search ellipse width.
    """
    evidence: list[Evidence] = []
    dx = com_x - lkp_x
    dz = com_z - lkp_z
    drift_len = math.hypot(dx, dz)
    if drift_len < 1.0:
        return evidence

    perp_x = -dz / drift_len
    perp_z = dx / drift_len
    drift_heading = math.atan2(dx, dz)
    perp_scale = max(80.0, leeway_uncertainty * 0.3)
    next_id = 0

    # Three debris points along the drift line — age decreases downstream.
    for frac, age in [(0.20, 2.5), (0.50, 1.5), (0.80, 0.5)]:
        base_x = lkp_x + dx * frac
        base_z = lkp_z + dz * frac
        jitter = float(rng.uniform(-perp_scale, perp_scale))
        ex = base_x + perp_x * jitter
        ez = base_z + perp_z * jitter
        ex, ez = _clamp_to_world(ex, ez, world_size, margin=30.0)
        evidence.append(
            Evidence(
                id=next_id,
                position=Vec3(ex, 0.0, ez),
                kind=EvidenceKind.DEBRIS.value,
                confidence=0.5,
                heading=None,
                age_hours=age,
            )
        )
        next_id += 1

    # Directional "slick" (footprint kind, heading = current bearing) near
    # the midpoint — tells the posterior to bias updates along-current.
    slick_x = lkp_x + dx * 0.55 + perp_x * float(rng.uniform(-60.0, 60.0))
    slick_z = lkp_z + dz * 0.55 + perp_z * float(rng.uniform(-60.0, 60.0))
    slick_x, slick_z = _clamp_to_world(slick_x, slick_z, world_size, margin=30.0)
    evidence.append(
        Evidence(
            id=next_id,
            position=Vec3(slick_x, 0.0, slick_z),
            kind=EvidenceKind.FOOTPRINT.value,
            confidence=0.55,
            heading=drift_heading,
            age_hours=1.0,
        )
    )
    next_id += 1

    # Signal marker at the projected drift COM (flare, dye, or EPIRB).
    sf_x = com_x + float(rng.uniform(-120.0, 120.0))
    sf_z = com_z + float(rng.uniform(-120.0, 120.0))
    sf_x, sf_z = _clamp_to_world(sf_x, sf_z, world_size, margin=30.0)
    evidence.append(
        Evidence(
            id=next_id,
            position=Vec3(sf_x, 0.0, sf_z),
            kind=EvidenceKind.SIGNAL_FIRE.value,
            confidence=0.9,
            heading=None,
            age_hours=0.25,
        )
    )

    return evidence


def _plant_runout_evidence(
    fracture_x: float,
    fracture_z: float,
    toe_x: float,
    toe_z: float,
    world_size: int,
    rng: np.random.Generator,
    *,
    half_angle_deg: float = 18.0,
) -> list[Evidence]:
    """Avalanche runout cone — equipment strewn from fracture line to toe.

    Geometry is a widening cone along the fall-line:
      * fracture-line footprint pointing down-slope (ski/snowboard tracks)
      * three debris points fanning out, lateral spread proportional to
        cone half-angle × distance from fracture
      * signal fire near the toe — strongest burial concentration, often
        where an avalanche-airbag beacon would be active

    Evidence is fresh (sub-hour ages) — avalanche survival window is
    short so clues are expected to be recent.
    """
    evidence: list[Evidence] = []
    dx = toe_x - fracture_x
    dz = toe_z - fracture_z
    runout = math.hypot(dx, dz)
    if runout < 1.0:
        return evidence

    perp_x = -dz / runout
    perp_z = dx / runout
    fall_heading = math.atan2(dx, dz)
    half_angle = math.radians(half_angle_deg)
    next_id = 0

    # Fracture-line footprint — directional, down-slope.
    evidence.append(
        Evidence(
            id=next_id,
            position=Vec3(fracture_x, 0.0, fracture_z),
            kind=EvidenceKind.FOOTPRINT.value,
            confidence=0.65,
            heading=fall_heading,
            age_hours=0.5,
        )
    )
    next_id += 1

    # Three debris points along the cone, alternating lateral side with
    # spread proportional to cone half-angle × distance.
    cone_layout = [(0.40, -0.6), (0.65, 0.7), (0.90, -0.3)]
    for frac, side in cone_layout:
        base_x = fracture_x + dx * frac
        base_z = fracture_z + dz * frac
        max_spread = math.tan(half_angle) * runout * frac
        lateral = max_spread * side
        lateral += float(rng.uniform(-30.0, 30.0))
        ex = base_x + perp_x * lateral
        ez = base_z + perp_z * lateral
        ex, ez = _clamp_to_world(ex, ez, world_size, margin=30.0)
        evidence.append(
            Evidence(
                id=next_id,
                position=Vec3(ex, 0.0, ez),
                kind=EvidenceKind.DEBRIS.value,
                confidence=0.5,
                heading=None,
                age_hours=0.5,
            )
        )
        next_id += 1

    # Signal fire near the toe — avalanche-airbag beacon / freshest signal.
    sf_x = toe_x + float(rng.uniform(-100.0, 100.0))
    sf_z = toe_z + float(rng.uniform(-100.0, 100.0))
    sf_x, sf_z = _clamp_to_world(sf_x, sf_z, world_size, margin=30.0)
    evidence.append(
        Evidence(
            id=next_id,
            position=Vec3(sf_x, 0.0, sf_z),
            kind=EvidenceKind.SIGNAL_FIRE.value,
            confidence=0.85,
            heading=None,
            age_hours=0.1,
        )
    )

    return evidence


def _plant_structure_evidence(
    clusters: list[Cluster],
    world_size: int,
    rng: np.random.Generator,
) -> list[Evidence]:
    """Disaster response — evidence concentrated at densest-structure clusters.

    Clusters are weighted by structural density; rather than lay a single
    trail, we plant:
      * signal fires at the two highest-weight clusters (trapped survivors
        signal from the most-populated collapse sites)
      * debris at the next three clusters (secondary collapse sites)

    Jitter is bounded by each cluster's own radius, so evidence sits inside
    the cluster footprint (where survivors actually are) rather than between
    them.
    """
    if not clusters:
        return []

    ranked = sorted(clusters, key=lambda c: c[3], reverse=True)
    evidence: list[Evidence] = []
    next_id = 0

    for cx, cz, _cr, _w in ranked[:2]:
        jx = float(rng.uniform(-80.0, 80.0))
        jz = float(rng.uniform(-80.0, 80.0))
        ex, ez = _clamp_to_world(cx + jx, cz + jz, world_size, margin=30.0)
        evidence.append(
            Evidence(
                id=next_id,
                position=Vec3(ex, 0.0, ez),
                kind=EvidenceKind.SIGNAL_FIRE.value,
                confidence=0.85,
                heading=None,
                age_hours=0.5,
            )
        )
        next_id += 1

    for cx, cz, cr, _w in ranked[2:5]:
        spread = max(40.0, cr * 0.3)
        jx = float(rng.uniform(-spread, spread))
        jz = float(rng.uniform(-spread, spread))
        ex, ez = _clamp_to_world(cx + jx, cz + jz, world_size, margin=30.0)
        evidence.append(
            Evidence(
                id=next_id,
                position=Vec3(ex, 0.0, ez),
                kind=EvidenceKind.DEBRIS.value,
                confidence=0.55,
                heading=None,
                age_hours=1.5,
            )
        )
        next_id += 1

    return evidence


# ---------------------------------------------------------------------------
# Mission templates
# ---------------------------------------------------------------------------


def aircraft_crash(world_size: int, seed: int) -> SearchMission:
    """Small aircraft down in mountainous/forested terrain.

    Last radar contact, heading and airspeed known. Debris field stretches
    along the inferred flight path. Some occupants likely walked away seeking
    help.
    """
    rng = np.random.default_rng(seed + 1001)

    # Last radar contact — somewhere in the world's interior so the cone fits.
    last_radar_x = world_size * float(rng.uniform(0.3, 0.55))
    last_radar_z = world_size * float(rng.uniform(0.3, 0.55))

    bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    flight_length = world_size * float(rng.uniform(0.20, 0.32))

    # Impact site is at the far end of the inferred flight cone.
    crash_x = last_radar_x + math.cos(bearing) * flight_length
    crash_z = last_radar_z + math.sin(bearing) * flight_length
    crash_x, crash_z = _clamp_to_world(crash_x, crash_z, world_size)

    clusters: list[Cluster] = []

    # 1. Impact site — tight cluster, 40% of survivors.
    clusters.append((crash_x, crash_z, 150.0, 0.40))

    # 2. Debris field along flight path (back toward last radar).
    debris_bearing = bearing + math.pi
    for i in range(3):
        offset = 300 + i * 400
        dx = crash_x + math.cos(debris_bearing) * offset
        dz = crash_z + math.sin(debris_bearing) * offset
        dx, dz = _clamp_to_world(dx, dz, world_size)
        clusters.append((dx, dz, 100.0 + i * 30, 0.08))

    # 3. Walkaways — survivors who left the impact site seeking help.
    for _ in range(2):
        walk_angle = bearing + float(rng.uniform(-1.0, 1.0))
        walk_dist = float(rng.uniform(800, 2500))
        wx = crash_x + math.cos(walk_angle) * walk_dist
        wz = crash_z + math.sin(walk_angle) * walk_dist
        wx, wz = _clamp_to_world(wx, wz, world_size)
        clusters.append((wx, wz, 300.0, 0.08))

    # Base: 40% of mass is at the impact site, with debris trailing back along
    # the flight path. Stage the base ~600m short of the impact site so the
    # impact, the near debris segments, and the inner walkaways are all within
    # fresh-drone round-trip range. The far debris/radar end is left for
    # second-cycle drones (Phase 8 will replace this with FOBs).
    standoff_m = 600.0
    base_x = crash_x - math.cos(bearing) * standoff_m
    base_z = crash_z - math.sin(bearing) * standoff_m
    base_x, base_z = _clamp_to_world(base_x, base_z, world_size, margin=100.0)

    intel_pins = [
        {"label": "Last radar contact", "kind": "radar", "position": [last_radar_x, last_radar_z]},
        {"label": "Inferred impact area", "kind": "impact", "position": [crash_x, crash_z]},
    ]

    # Evidence trail: from the last-radar position along the flight path
    # toward the impact. Debris strewn between, signal fire near impact
    # where walking survivors would gather.
    evidence_rng = np.random.default_rng(seed + 2001)
    evidence = _plant_trail_evidence(
        start_x=last_radar_x,
        start_z=last_radar_z,
        end_x=crash_x,
        end_z=crash_z,
        world_size=world_size,
        rng=evidence_rng,
        signal_fire_at_end=True,
    )

    return SearchMission(
        name="aircraft_crash",
        title="Aircraft Crash — Mountain SAR",
        description=(
            "Single-engine aircraft lost contact with ATC. Last radar return placed it "
            "on a steady heading at cruise altitude. Likely impact lies along the "
            "projected flight path; survivors may have walked away from the wreckage."
        ),
        known_facts=[
            f"Last radar contact at ({last_radar_x:.0f}, {last_radar_z:.0f})",
            f"Heading {math.degrees(bearing):.0f}°, debris cone {flight_length:.0f}m long",
            "Survivor count unknown (1-8 estimated)",
            "Impact + walkaways expected",
        ],
        base_position=Vec3(base_x, 0.0, base_z),
        survivor_count_range=(3, 8),
        survival_window_seconds=4.0 * 3600.0,
        clusters=clusters,
        world_size=world_size,
        seed=seed,
        intel_pins=intel_pins,
        evidence=evidence,
    )


def lost_hiker(world_size: int, seed: int) -> SearchMission:
    """Solo hiker overdue from a day trip.

    Trailhead known. Last cell-phone ping placed them on the trail several
    hours ago. Likely radius is bounded by max walking speed × elapsed time,
    biased downhill (drainage pull) and along the trail.
    """
    rng = np.random.default_rng(seed + 1002)

    # Trailhead — base sits here, real SAR command posts almost always do.
    trailhead_x = world_size * float(rng.uniform(0.2, 0.4))
    trailhead_z = world_size * float(rng.uniform(0.2, 0.4))

    # Last ping along the trail bearing.
    trail_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    last_ping_dist = float(rng.uniform(400.0, 900.0))
    last_ping_x = trailhead_x + math.cos(trail_bearing) * last_ping_dist
    last_ping_z = trailhead_z + math.sin(trail_bearing) * last_ping_dist
    last_ping_x, last_ping_z = _clamp_to_world(last_ping_x, last_ping_z, world_size)

    # Time-since-ping × walking speed defines the maximum-reach circle.
    hours_overdue = float(rng.uniform(2.0, 5.0))
    max_walk_speed_mps = 0.6
    reach_radius = hours_overdue * 3600.0 * max_walk_speed_mps  # meters

    clusters: list[Cluster] = []

    # 1. Hiker is most likely along the trail past the last ping — elongated
    # ellipse along the trail bearing, falling off behind. Three clusters
    # in a line along the bearing approximate this.
    for i, (dist_factor, weight) in enumerate([(0.3, 0.25), (0.6, 0.35), (0.9, 0.15)]):
        d = reach_radius * dist_factor
        cx = last_ping_x + math.cos(trail_bearing) * d
        cz = last_ping_z + math.sin(trail_bearing) * d
        cx, cz = _clamp_to_world(cx, cz, world_size)
        clusters.append((cx, cz, 250.0 + i * 100, weight))

    # 2. Drainage pull — survivors lost in mountains often head downhill seeking
    # water/road. Add a cluster perpendicular-ish to the trail bearing to
    # represent that bias.
    drain_bearing = trail_bearing + math.pi / 2 + float(rng.uniform(-0.3, 0.3))
    drain_dist = reach_radius * 0.5
    dcx = last_ping_x + math.cos(drain_bearing) * drain_dist
    dcz = last_ping_z + math.sin(drain_bearing) * drain_dist
    dcx, dcz = _clamp_to_world(dcx, dcz, world_size)
    clusters.append((dcx, dcz, 350.0, 0.20))

    # 3. Off-trail wandering — small probability the hiker reversed course.
    back_dist = reach_radius * 0.25
    bcx = last_ping_x - math.cos(trail_bearing) * back_dist
    bcz = last_ping_z - math.sin(trail_bearing) * back_dist
    bcx, bcz = _clamp_to_world(bcx, bcz, world_size)
    clusters.append((bcx, bcz, 200.0, 0.05))

    # Base: forward operating post. Real SAR teams stage near the last-known
    # position when the trailhead is hours away by foot. We pull the base
    # ~30% along the prior's trail-bearing axis so the reach circle's near
    # half is round-trippable. Trailhead becomes an intel pin, not the base.
    forward_offset = reach_radius * 0.4
    base_x = last_ping_x + math.cos(trail_bearing) * forward_offset
    base_z = last_ping_z + math.sin(trail_bearing) * forward_offset
    base_x, base_z = _clamp_to_world(base_x, base_z, world_size, margin=100.0)

    intel_pins = [
        {"label": "Trailhead", "kind": "trailhead", "position": [trailhead_x, trailhead_z]},
        {"label": "Last cell ping", "kind": "ping", "position": [last_ping_x, last_ping_z]},
        {"label": "Forward command post", "kind": "fcp", "position": [base_x, base_z]},
    ]

    # Evidence trail: from last cell ping along the trail bearing to the
    # mid-reach cluster (the 0.6 reach-factor one, which has the highest
    # weight). Footprints point forward along the trail.
    primary_cluster_d = reach_radius * 0.6
    primary_cx = last_ping_x + math.cos(trail_bearing) * primary_cluster_d
    primary_cz = last_ping_z + math.sin(trail_bearing) * primary_cluster_d
    primary_cx, primary_cz = _clamp_to_world(primary_cx, primary_cz, world_size)
    evidence_rng = np.random.default_rng(seed + 2002)
    evidence = _plant_trail_evidence(
        start_x=last_ping_x,
        start_z=last_ping_z,
        end_x=primary_cx,
        end_z=primary_cz,
        world_size=world_size,
        rng=evidence_rng,
        signal_fire_at_end=True,
    )

    return SearchMission(
        name="lost_hiker",
        title="Lost Hiker — Wilderness SAR",
        description=(
            "Solo hiker overdue from a day trip. Last cell tower ping placed them on "
            "the trail several hours ago. Forward command post staged near the last "
            "ping; max-reach circle bounded by walking speed and biased along the "
            "trail and downhill drainage."
        ),
        known_facts=[
            f"Trailhead at ({trailhead_x:.0f}, {trailhead_z:.0f})",
            f"Last ping {hours_overdue:.1f}h ago at ({last_ping_x:.0f}, {last_ping_z:.0f})",
            f"Max-reach circle ~{reach_radius:.0f}m from last ping",
            "1 subject expected, possibly disoriented",
        ],
        base_position=Vec3(base_x, 0.0, base_z),
        survivor_count_range=(1, 1),
        survival_window_seconds=24.0 * 3600.0,
        clusters=clusters,
        world_size=world_size,
        seed=seed,
        intel_pins=intel_pins,
        evidence=evidence,
    )


def maritime_sar(world_size: int, seed: int) -> SearchMission:
    """Person/vessel in the water, drifting with current and wind.

    Last-known-position fixed (often from EPIRB or AIS). Drift since contact
    is the dominant factor — the search ellipse is offset downstream and
    elongated along the drift vector.
    """
    rng = np.random.default_rng(seed + 1003)

    # Last known position — central so drift can carry it downstream without
    # immediately leaving the world.
    lkp_x = world_size * float(rng.uniform(0.3, 0.5))
    lkp_z = world_size * float(rng.uniform(0.3, 0.5))

    # Current vector (m/s) — surface current dominates over wind at sea.
    current_speed = float(rng.uniform(0.3, 0.6))
    current_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    cx_v = math.cos(current_bearing) * current_speed
    cz_v = math.sin(current_bearing) * current_speed

    hours_elapsed = float(rng.uniform(1.5, 4.0))
    drift_seconds = hours_elapsed * 3600.0
    drift_x = cx_v * drift_seconds
    drift_z = cz_v * drift_seconds

    # Center-of-mass of the prior is offset downstream from the LKP by drift.
    com_x = lkp_x + drift_x
    com_z = lkp_z + drift_z
    com_x, com_z = _clamp_to_world(com_x, com_z, world_size)

    # Drift uncertainty grows linearly with time. Real maritime SAR uses
    # 5-15% of drift distance per hour as the leeway uncertainty.
    drift_dist = math.sqrt(drift_x * drift_x + drift_z * drift_z)
    leeway_uncertainty = max(150.0, drift_dist * 0.25)

    clusters: list[Cluster] = []

    # 1. Main drift cluster at projected current position.
    clusters.append((com_x, com_z, leeway_uncertainty, 0.60))

    # 2. A pair of trailing clusters back along the drift line — wreckage
    # debris fans, slower-drifting bodies, etc.
    for i, weight in enumerate([0.15, 0.10]):
        back_dist = drift_dist * (0.4 + i * 0.3)
        bx = com_x - math.cos(current_bearing) * back_dist
        bz = com_z - math.sin(current_bearing) * back_dist
        bx, bz = _clamp_to_world(bx, bz, world_size)
        clusters.append((bx, bz, leeway_uncertainty * (0.8 + i * 0.3), weight))

    # 3. Spread perpendicular to drift — wind leeway divergence.
    for sign in (-1.0, 1.0):
        perp = current_bearing + sign * math.pi / 2.0
        px = com_x + math.cos(perp) * leeway_uncertainty * 1.5
        pz = com_z + math.sin(perp) * leeway_uncertainty * 1.5
        px, pz = _clamp_to_world(px, pz, world_size)
        clusters.append((px, pz, leeway_uncertainty * 1.2, 0.075))

    # Base: a search vessel positioned at the leading edge of the drift
    # ellipse — close enough that drones can sweep both forward (extrapolated
    # drift) and backward (debris trail).
    base_x, base_z = _clamp_to_world(com_x, com_z, world_size, margin=100.0)

    intel_pins = [
        {"label": "Last known position", "kind": "lkp", "position": [lkp_x, lkp_z]},
        {"label": "Projected drift", "kind": "drift", "position": [com_x, com_z]},
    ]

    # Evidence trail: debris strewn along the drift line from LKP to the
    # projected COM, with a signal marker at the leading edge.
    evidence_rng = np.random.default_rng(seed + 2003)
    evidence = _plant_drift_evidence(
        lkp_x=lkp_x,
        lkp_z=lkp_z,
        com_x=com_x,
        com_z=com_z,
        leeway_uncertainty=leeway_uncertainty,
        world_size=world_size,
        rng=evidence_rng,
    )

    return SearchMission(
        name="maritime_sar",
        title="Maritime SAR — Drift Search",
        description=(
            "Vessel/person in water with EPIRB activation. Last known position fixed; "
            "current and wind drift estimated. Search ellipse offset downstream from "
            "LKP, elongated along the drift vector."
        ),
        known_facts=[
            f"LKP at ({lkp_x:.0f}, {lkp_z:.0f})",
            (
                f"{hours_elapsed:.1f}h elapsed; drift {drift_dist:.0f}m "
                f"bearing {math.degrees(current_bearing):.0f}°"
            ),
            f"Current {current_speed:.2f} m/s",
            "1-3 subjects, hypothermia clock running",
        ],
        base_position=Vec3(base_x, 0.0, base_z),
        survivor_count_range=(1, 3),
        survival_window_seconds=2.0 * 3600.0,
        clusters=clusters,
        world_size=world_size,
        seed=seed,
        intel_pins=intel_pins,
        evidence=evidence,
    )


def avalanche(world_size: int, seed: int) -> SearchMission:
    """Avalanche burial — fracture line above, runout fan below.

    Burials concentrate on the runout fan with a depth distribution (deeper
    near the toe, shallower near the sides). Survival window is short.
    """
    rng = np.random.default_rng(seed + 1004)

    # Fracture line at the top of a slope.
    fracture_x = world_size * float(rng.uniform(0.35, 0.55))
    fracture_z = world_size * float(rng.uniform(0.35, 0.55))

    # Slope bearing — fall line direction.
    slope_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    runout_length = float(rng.uniform(600.0, 1100.0))

    # Apex of the runout fan is just below the fracture; toe is at the far end.
    apex_x = fracture_x + math.cos(slope_bearing) * (runout_length * 0.2)
    apex_z = fracture_z + math.sin(slope_bearing) * (runout_length * 0.2)
    toe_x = fracture_x + math.cos(slope_bearing) * runout_length
    toe_z = fracture_z + math.sin(slope_bearing) * runout_length
    apex_x, apex_z = _clamp_to_world(apex_x, apex_z, world_size, margin=100.0)
    toe_x, toe_z = _clamp_to_world(toe_x, toe_z, world_size, margin=100.0)

    clusters: list[Cluster] = []

    # 1. Toe of the runout — most burials here (debris piles up).
    clusters.append((toe_x, toe_z, 200.0, 0.50))

    # 2. Mid-fan — secondary concentration.
    mid_x = (apex_x + toe_x) / 2.0
    mid_z = (apex_z + toe_z) / 2.0
    clusters.append((mid_x, mid_z, 250.0, 0.30))

    # 3. Lateral spread — survivors tossed sideways.
    for sign in (-1.0, 1.0):
        perp = slope_bearing + sign * math.pi / 2.0
        sx = mid_x + math.cos(perp) * 200.0
        sz = mid_z + math.sin(perp) * 200.0
        sx, sz = _clamp_to_world(sx, sz, world_size)
        clusters.append((sx, sz, 150.0, 0.10))

    # Base: at the fan apex (just below fracture line) — lets the swarm work
    # the entire runout downhill. Real-world avalanche teams also stage above.
    base_x, base_z = _clamp_to_world(apex_x, apex_z, world_size, margin=100.0)

    intel_pins = [
        {"label": "Fracture line", "kind": "fracture", "position": [fracture_x, fracture_z]},
        {"label": "Runout toe", "kind": "toe", "position": [toe_x, toe_z]},
    ]

    # Evidence trail: fracture-line track + debris fanning down the runout
    # cone + signal near the toe (highest burial concentration).
    evidence_rng = np.random.default_rng(seed + 2004)
    evidence = _plant_runout_evidence(
        fracture_x=fracture_x,
        fracture_z=fracture_z,
        toe_x=toe_x,
        toe_z=toe_z,
        world_size=world_size,
        rng=evidence_rng,
    )

    return SearchMission(
        name="avalanche",
        title="Avalanche — Burial Search",
        description=(
            "Reported avalanche with witness sighting of the fracture line. Burials "
            "concentrate on the runout fan, deeper toward the toe. Critical window: "
            "first 15-30 minutes for live recovery."
        ),
        known_facts=[
            f"Fracture line at ({fracture_x:.0f}, {fracture_z:.0f})",
            f"Runout {runout_length:.0f}m bearing {math.degrees(slope_bearing):.0f}°",
            f"Toe at ({toe_x:.0f}, {toe_z:.0f})",
            "2-6 subjects, partial burial likely",
        ],
        base_position=Vec3(base_x, 0.0, base_z),
        survivor_count_range=(2, 6),
        survival_window_seconds=30.0 * 60.0,
        clusters=clusters,
        world_size=world_size,
        seed=seed,
        intel_pins=intel_pins,
        evidence=evidence,
    )


def disaster_response(world_size: int, seed: int) -> SearchMission:
    """Wide-area disaster — earthquake, flood, structural collapse.

    Survivors distributed over an affected polygon, weighted by structure
    density (urban areas dominate). Less concentrated than the other
    scenarios but covering more ground.
    """
    rng = np.random.default_rng(seed + 1005)

    # Affected polygon centered somewhere in the world's interior.
    cx = world_size * float(rng.uniform(0.35, 0.55))
    cz = world_size * float(rng.uniform(0.35, 0.55))

    # Affected radius — a sizable chunk of the world.
    affected_radius = world_size * 0.15

    clusters: list[Cluster] = []

    # 6 clusters arranged around the affected center, mimicking
    # neighborhood-level concentrations of trapped survivors. Weights vary
    # to reflect structural-density variation.
    n_clusters = 6
    weights = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10]
    for i in range(n_clusters):
        angle = i * (2.0 * math.pi / n_clusters) + float(rng.uniform(-0.4, 0.4))
        dist = affected_radius * float(rng.uniform(0.3, 0.95))
        ex = cx + math.cos(angle) * dist
        ez = cz + math.sin(angle) * dist
        ex, ez = _clamp_to_world(ex, ez, world_size, margin=100.0)
        clusters.append((ex, ez, 350.0 + i * 50, weights[i]))

    # Base: command post at the affected-area centroid (typical disaster
    # response staging — close to the impact zone but not inside it).
    base_x, base_z = _clamp_to_world(cx, cz, world_size, margin=100.0)

    intel_pins = [
        {"label": "Affected centroid (base)", "kind": "centroid", "position": [cx, cz]},
        {
            "label": "Affected radius",
            "kind": "radius",
            "position": [cx, cz],
            "radius": affected_radius,
        },
    ]

    # Evidence: signal fires at the top-weight collapse clusters, debris
    # at the next tier. Anchored to cluster centers so clues sit inside
    # the same structure-density zones where survivors actually spawn.
    evidence_rng = np.random.default_rng(seed + 2005)
    evidence = _plant_structure_evidence(
        clusters=clusters,
        world_size=world_size,
        rng=evidence_rng,
    )

    return SearchMission(
        name="disaster_response",
        title="Disaster Response — Wide-Area Search",
        description=(
            "Major event reported across multiple neighborhoods. Survivors trapped or "
            "displaced within the affected zone. Search density biased toward higher "
            "structure density and known shelters."
        ),
        known_facts=[
            f"Impact centroid at ({cx:.0f}, {cz:.0f})",
            f"Affected radius ~{affected_radius:.0f}m",
            "20-40 subjects estimated across the zone",
            "Multiple structural collapse sites likely",
        ],
        base_position=Vec3(base_x, 0.0, base_z),
        survivor_count_range=(20, 40),
        survival_window_seconds=12.0 * 3600.0,
        clusters=clusters,
        world_size=world_size,
        seed=seed,
        intel_pins=intel_pins,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


MISSION_FACTORIES = {
    "aircraft_crash": aircraft_crash,
    "lost_hiker": lost_hiker,
    "maritime_sar": maritime_sar,
    "avalanche": avalanche,
    "disaster_response": disaster_response,
}


def build_mission(name: str, world_size: int, seed: int) -> SearchMission:
    """Construct a mission by name. Falls back to aircraft_crash if unknown."""
    factory = MISSION_FACTORIES.get(name, aircraft_crash)
    return factory(world_size, seed)


def available_missions() -> list[str]:
    """List of registered mission names — used by the frontend selector."""
    return list(MISSION_FACTORIES.keys())
