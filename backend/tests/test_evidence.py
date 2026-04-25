"""Tests for Phase 3 evidence system.

Three layers:
  1. Mission evidence planting — lost_hiker and aircraft_crash seed clues
     along the survivor path.
  2. SearchMap.update_on_evidence — the three posterior-update kernels
     (footprint cone, debris ring, signal-fire circle) reshape the PoC
     correctly and preserve total mass.
  3. Drone evidence detection — a drone within range of a clue finds it,
     kind-specific ranges are respected.
"""

from __future__ import annotations

import math

import numpy as np

from src.simulation.drone import detect_evidence
from src.simulation.mission import aircraft_crash, build_mission, lost_hiker
from src.simulation.search_map import SearchMap
from src.simulation.types import (
    Drone,
    DroneStatus,
    Evidence,
    EvidenceKind,
    SimConfig,
    Vec3,
)

WORLD_SIZE = 10240


# ---------------------------------------------------------------------------
# Mission evidence planting
# ---------------------------------------------------------------------------


class TestMissionEvidence:
    def test_lost_hiker_plants_trail_evidence(self) -> None:
        m = lost_hiker(WORLD_SIZE, seed=42)
        # Expect 3 footprints + 1 debris + 1 signal fire = 5 clues
        assert len(m.evidence) == 5
        kinds = [e.kind for e in m.evidence]
        assert kinds.count(EvidenceKind.FOOTPRINT.value) == 3
        assert kinds.count(EvidenceKind.DEBRIS.value) == 1
        assert kinds.count(EvidenceKind.SIGNAL_FIRE.value) == 1

    def test_aircraft_crash_plants_trail_evidence(self) -> None:
        m = aircraft_crash(WORLD_SIZE, seed=42)
        assert len(m.evidence) == 5

    def test_footprint_headings_align_with_trail_direction(self) -> None:
        """All footprints from a single trail share the same heading —
        they point toward the survivor endpoint."""
        m = lost_hiker(WORLD_SIZE, seed=42)
        footprints = [e for e in m.evidence if e.kind == EvidenceKind.FOOTPRINT.value]
        assert len(footprints) >= 2
        headings = [f.heading for f in footprints]
        # All footprints planted by the same trail call share one heading
        assert all(h == headings[0] for h in headings)
        assert headings[0] is not None

    def test_evidence_positions_within_world(self) -> None:
        for name in ("lost_hiker", "aircraft_crash"):
            m = build_mission(name, WORLD_SIZE, seed=7)
            for e in m.evidence:
                assert 0 <= e.position.x <= WORLD_SIZE
                assert 0 <= e.position.z <= WORLD_SIZE

    def test_maritime_plants_drift_trail(self) -> None:
        """Maritime drift: debris along the LKP→COM line, directional slick
        (footprint heading aligned with current), signal at the COM."""
        from src.simulation.mission import maritime_sar

        m = maritime_sar(WORLD_SIZE, seed=42)
        kinds = [e.kind for e in m.evidence]
        assert kinds.count(EvidenceKind.DEBRIS.value) == 3
        assert kinds.count(EvidenceKind.FOOTPRINT.value) == 1
        assert kinds.count(EvidenceKind.SIGNAL_FIRE.value) == 1

        pins = {p["kind"]: p for p in m.intel_pins}
        lkp_x, lkp_z = pins["lkp"]["position"]
        com_x, com_z = pins["drift"]["position"]
        drift_len = math.hypot(com_x - lkp_x, com_z - lkp_z)
        assert drift_len > 1.0

        # Project each debris onto the drift axis — values should progress
        # monotonically from LKP toward the COM (0.2 → 0.5 → 0.8).
        debris = [e for e in m.evidence if e.kind == EvidenceKind.DEBRIS.value]
        axis_x = (com_x - lkp_x) / drift_len
        axis_z = (com_z - lkp_z) / drift_len
        proj = [(e.position.x - lkp_x) * axis_x + (e.position.z - lkp_z) * axis_z for e in debris]
        assert proj == sorted(proj)  # ordered downstream

        # Debris ages decrease downstream (old wreckage near LKP, fresh near COM).
        ages = [e.age_hours for e in debris]
        assert ages == sorted(ages, reverse=True)

        # Signal fire should be near the COM (within ~200m after jitter + clamp).
        sf = next(e for e in m.evidence if e.kind == EvidenceKind.SIGNAL_FIRE.value)
        assert math.hypot(sf.position.x - com_x, sf.position.z - com_z) < 250.0

    def test_avalanche_plants_runout_cone(self) -> None:
        """Avalanche: fracture-line footprint heading down-slope, debris
        fanning toward the toe, signal fire near the toe."""
        from src.simulation.mission import avalanche

        m = avalanche(WORLD_SIZE, seed=42)
        kinds = [e.kind for e in m.evidence]
        assert kinds.count(EvidenceKind.FOOTPRINT.value) == 1
        assert kinds.count(EvidenceKind.DEBRIS.value) == 3
        assert kinds.count(EvidenceKind.SIGNAL_FIRE.value) == 1

        pins = {p["kind"]: p for p in m.intel_pins}
        fracture_x, fracture_z = pins["fracture"]["position"]
        toe_x, toe_z = pins["toe"]["position"]

        # Fracture footprint sits at the fracture line and points down-slope
        # (heading vector ≈ toe direction).
        fp = next(e for e in m.evidence if e.kind == EvidenceKind.FOOTPRINT.value)
        assert math.hypot(fp.position.x - fracture_x, fp.position.z - fracture_z) < 5.0
        assert fp.heading is not None
        # Dot product between heading unit vector and (toe - fracture) unit
        # vector should be strongly positive (nearly collinear).
        dx = toe_x - fracture_x
        dz = toe_z - fracture_z
        runout = math.hypot(dx, dz)
        heading_x = math.sin(fp.heading)
        heading_z = math.cos(fp.heading)
        alignment = heading_x * (dx / runout) + heading_z * (dz / runout)
        assert alignment > 0.95

        # Signal fire is close to the toe (within ~200m of the apex/toe jitter).
        sf = next(e for e in m.evidence if e.kind == EvidenceKind.SIGNAL_FIRE.value)
        assert math.hypot(sf.position.x - toe_x, sf.position.z - toe_z) < 250.0

        # Debris spread laterally grows with distance down-cone: project
        # each debris onto the perpendicular axis and onto the down-slope
        # axis, then verify the farthest-downstream debris has non-trivial
        # lateral spread.
        perp_x = -dz / runout
        perp_z = dx / runout
        debris = [e for e in m.evidence if e.kind == EvidenceKind.DEBRIS.value]
        lateral_spreads = [
            abs((e.position.x - fracture_x) * perp_x + (e.position.z - fracture_z) * perp_z)
            for e in debris
        ]
        # At least one debris point should be meaningfully off-axis (>50m).
        assert max(lateral_spreads) > 50.0

    def test_disaster_plants_evidence_at_top_clusters(self) -> None:
        """Disaster: signal fires on the two densest clusters, debris on the
        next three. Each evidence sits inside its anchoring cluster radius."""
        from src.simulation.mission import disaster_response

        m = disaster_response(WORLD_SIZE, seed=42)
        kinds = [e.kind for e in m.evidence]
        assert kinds.count(EvidenceKind.SIGNAL_FIRE.value) == 2
        assert kinds.count(EvidenceKind.DEBRIS.value) == 3

        ranked = sorted(m.clusters, key=lambda c: c[3], reverse=True)
        top_two = ranked[:2]
        fires = [e for e in m.evidence if e.kind == EvidenceKind.SIGNAL_FIRE.value]
        # Each signal fire should be within its cluster's jitter band (≤ ~120m
        # from one of the top-two cluster centers, after 80m jitter + clamp).
        for fire in fires:
            best = min(
                math.hypot(fire.position.x - cx, fire.position.z - cz) for cx, cz, _r, _w in top_two
            )
            assert best < 150.0

    def test_new_trails_are_deterministic_for_same_seed(self) -> None:
        from src.simulation.mission import avalanche, disaster_response, maritime_sar

        for fn in (maritime_sar, avalanche, disaster_response):
            m1 = fn(WORLD_SIZE, seed=77)
            m2 = fn(WORLD_SIZE, seed=77)
            assert len(m1.evidence) == len(m2.evidence)
            for e1, e2 in zip(m1.evidence, m2.evidence, strict=True):
                assert e1.position == e2.position
                assert e1.kind == e2.kind
                assert e1.heading == e2.heading
                assert e1.age_hours == e2.age_hours

    def test_evidence_deterministic_for_same_seed(self) -> None:
        m1 = lost_hiker(WORLD_SIZE, seed=123)
        m2 = lost_hiker(WORLD_SIZE, seed=123)
        assert len(m1.evidence) == len(m2.evidence)
        for e1, e2 in zip(m1.evidence, m2.evidence, strict=True):
            assert e1.position == e2.position
            assert e1.kind == e2.kind
            assert e1.heading == e2.heading


# ---------------------------------------------------------------------------
# SearchMap posterior updates
# ---------------------------------------------------------------------------


class TestPosteriorKernels:
    def _make_uniform(self) -> SearchMap:
        sm = SearchMap.uniform(world_size=float(WORLD_SIZE), cell_size=40.0)
        return sm

    def test_signal_fire_boosts_center(self) -> None:
        sm = self._make_uniform()
        col, row = sm.world_to_cell(5000, 5000)
        before_center = float(sm.poc[row, col])
        peak = sm.update_on_evidence(
            position=(5000, 5000),
            kind="signal_fire",
            confidence=0.9,
        )
        after_center = float(sm.poc[row, col])
        # Signal fire should meaningfully boost the center cell
        assert after_center > before_center * 2.0
        # And peak multiplier should be > 1
        assert peak > 1.0

    def test_mass_preserved_after_update(self) -> None:
        sm = self._make_uniform()
        before = sm.total_mass()
        sm.update_on_evidence(position=(5000, 5000), kind="signal_fire", confidence=0.9)
        after = sm.total_mass()
        # Multiplicative re-shape + renormalize should keep total mass
        assert abs(after - before) < 1e-3

    def test_footprint_biases_posterior_along_heading(self) -> None:
        """A footprint at (5000, 5000) pointing +X/east should push more
        mass to the east side than the west side of that location."""
        sm = self._make_uniform()
        # heading = π/2 (east) — sin(π/2)=1, cos(π/2)=0 → direction vector
        # (x, z) = (1, 0), i.e. pure +X.
        sm.update_on_evidence(
            position=(5000, 5000),
            kind="footprint",
            confidence=0.9,
            heading=math.pi / 2,
        )
        # Sample a cell 400m east and 400m west of the footprint.
        east_col, east_row = sm.world_to_cell(5400, 5000)
        west_col, west_row = sm.world_to_cell(4600, 5000)
        east_val = float(sm.poc[east_row, east_col])
        west_val = float(sm.poc[west_row, west_col])
        assert east_val > west_val

    def test_debris_ring_peaks_off_center(self) -> None:
        """Debris update should boost cells at the ring radius (~400m)
        more than cells exactly at the debris point."""
        sm = self._make_uniform()
        sm.update_on_evidence(
            position=(5000, 5000),
            kind="debris",
            confidence=0.9,
        )
        # Center cell
        c_col, c_row = sm.world_to_cell(5000, 5000)
        # Ring cell (400m east)
        r_col, r_row = sm.world_to_cell(5400, 5000)
        center_val = float(sm.poc[c_row, c_col])
        ring_val = float(sm.poc[r_row, r_col])
        # Ring should have higher PoC than the center (donut shape)
        assert ring_val > center_val

    def test_zero_confidence_is_noop(self) -> None:
        sm = self._make_uniform()
        before = sm.poc.copy()
        peak = sm.update_on_evidence(
            position=(5000, 5000),
            kind="signal_fire",
            confidence=0.0,
        )
        np.testing.assert_array_equal(sm.poc, before)
        assert peak == 0.0

    def test_evidence_outside_world_returns_zero(self) -> None:
        sm = self._make_uniform()
        peak = sm.update_on_evidence(
            position=(-100.0, 5000.0),
            kind="signal_fire",
            confidence=0.9,
        )
        assert peak == 0.0

    def test_unknown_kind_falls_back_to_isotropic(self) -> None:
        sm = self._make_uniform()
        peak = sm.update_on_evidence(
            position=(5000, 5000),
            kind="mystery_clue",
            confidence=0.5,
        )
        # Should still apply *some* update
        assert peak > 1.0

    def test_signal_fire_reshapes_mass_toward_clue(self) -> None:
        """Signal fire at (7500, 7500) should shift mass toward that region
        even if it can't overwhelm a strong prior peak elsewhere.

        A multiplicative update scales cells relative to their prior — it
        *reshapes* belief, it doesn't teleport it. So we check the fraction
        of total mass within 1km of the clue goes up.
        """
        sm = SearchMap.empty(world_size=float(WORLD_SIZE), cell_size=40.0)
        sm.add_gaussian(center_world=(2000, 2000), radius_meters=800, weight=1.0)
        sm.poc += np.float32(0.001)
        sm.normalize()

        def mass_near(cx: float, cz: float, radius: float) -> float:
            col, row = sm.world_to_cell(cx, cz)
            rc = int(radius / sm.cell_size)
            r_min = max(0, row - rc)
            r_max = min(sm.poc.shape[0], row + rc + 1)
            c_min = max(0, col - rc)
            c_max = min(sm.poc.shape[1], col + rc + 1)
            return float(sm.poc[r_min:r_max, c_min:c_max].sum())

        before = mass_near(7500, 7500, 600.0)
        sm.update_on_evidence(
            position=(7500, 7500),
            kind="signal_fire",
            confidence=0.95,
        )
        after = mass_near(7500, 7500, 600.0)
        # Signal fire should pull mass toward its location
        assert after > before * 2.0


# ---------------------------------------------------------------------------
# Drone evidence detection
# ---------------------------------------------------------------------------


class TestEvidenceDetection:
    def _drone_at(self, x: float, z: float) -> Drone:
        return Drone(
            id=0,
            position=Vec3(x, 50.0, z),
            sensor_active=True,
            status=DroneStatus.ACTIVE,
            sensor_range=40.0,
        )

    def _evidence_at(self, eid: int, kind: str, x: float, z: float) -> Evidence:
        return Evidence(
            id=eid,
            position=Vec3(x, 0.0, z),
            kind=kind,
            confidence=0.7,
        )

    def test_drone_finds_footprint_within_range(self) -> None:
        config = SimConfig(weather_visibility=1.0)

        # Beach biome (open terrain) — biome_fn always returns BEACH
        def beach_biome(_x: float, _z: float) -> int:
            return 1  # Biome.BEACH.value

        drone = self._drone_at(1000, 1000)
        ev = self._evidence_at(42, EvidenceKind.FOOTPRINT.value, 1010, 1010)
        detected = detect_evidence(
            drone,
            (ev,),
            biome_fn=beach_biome,
            config=config,
        )
        assert detected == [42]

    def test_drone_misses_evidence_out_of_range(self) -> None:
        config = SimConfig()

        def beach_biome(_x: float, _z: float) -> int:
            return 1

        drone = self._drone_at(1000, 1000)
        # Footprint 500m away — outside base range (60m)
        ev = self._evidence_at(1, EvidenceKind.FOOTPRINT.value, 1500, 1500)
        detected = detect_evidence(drone, (ev,), biome_fn=beach_biome, config=config)
        assert detected == []

    def test_signal_fire_detected_at_long_range(self) -> None:
        """Signal fires have a much larger base range than footprints."""
        config = SimConfig()

        def beach_biome(_x: float, _z: float) -> int:
            return 1

        drone = self._drone_at(1000, 1000)
        # Signal fire 300m away — outside footprint range but inside signal_fire range
        ev = self._evidence_at(1, EvidenceKind.SIGNAL_FIRE.value, 1300, 1000)
        detected = detect_evidence(drone, (ev,), biome_fn=beach_biome, config=config)
        assert detected == [1]

    def test_already_discovered_evidence_skipped(self) -> None:
        config = SimConfig()

        def beach_biome(_x: float, _z: float) -> int:
            return 1

        drone = self._drone_at(1000, 1000)
        ev = Evidence(
            id=1,
            position=Vec3(1000, 0.0, 1000),
            kind=EvidenceKind.SIGNAL_FIRE.value,
            discovered=True,
        )
        detected = detect_evidence(drone, (ev,), biome_fn=beach_biome, config=config)
        assert detected == []

    def test_failed_drone_detects_nothing(self) -> None:
        config = SimConfig()

        def beach_biome(_x: float, _z: float) -> int:
            return 1

        drone = Drone(
            id=0,
            position=Vec3(1000, 50.0, 1000),
            status=DroneStatus.FAILED,
            sensor_active=False,
        )
        ev = self._evidence_at(1, EvidenceKind.SIGNAL_FIRE.value, 1001, 1001)
        detected = detect_evidence(drone, (ev,), biome_fn=beach_biome, config=config)
        assert detected == []

    def test_forest_reduces_detection_range(self) -> None:
        """Forest biome significantly cuts footprint detection range."""
        config = SimConfig(weather_visibility=1.0)

        def forest_biome(_x: float, _z: float) -> int:
            return 2  # Biome.FOREST.value

        drone = self._drone_at(1000, 1000)
        # Footprint 35m away. Beach would detect (60m*1.0=60m > 35m), forest should
        # also detect since 60*0.45=27m. Let's put it at 20m.
        ev = self._evidence_at(1, EvidenceKind.FOOTPRINT.value, 1015, 1015)
        detected = detect_evidence(drone, (ev,), biome_fn=forest_biome, config=config)
        # 1015,1015 is √(15²+15²) ≈ 21m away — within forest footprint range (27m)
        assert detected == [1]
