"""Tests for the extended MetricsTracker.

The old tracker only exposed coverage %, survivor count, and two weak
derived scores. Phase 3 added: MTTD, survival-window hit rate, per-
entity discovery timelines, PoC entropy, evidence→survivor latency,
drone-km flown, active fraction. These tests lock in the math.
"""

from __future__ import annotations

import math

import numpy as np

from src.simulation.metrics import MetricsTracker, _shannon_entropy
from src.simulation.search_map import SearchMap
from src.simulation.types import (
    FOG_UNEXPLORED,
    Drone,
    DroneStatus,
    Evidence,
    EvidenceKind,
    SimEvent,
    Survivor,
    Terrain,
    Vec3,
    WorldState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_world(
    *,
    tick: int = 0,
    elapsed: float = 0.0,
    drones: tuple[Drone, ...] = (),
    survivors: tuple[Survivor, ...] = (),
    evidence: tuple[Evidence, ...] = (),
    events: tuple[SimEvent, ...] = (),
    search_map: SearchMap | None = None,
) -> WorldState:
    fog = np.full((64, 64), FOG_UNEXPLORED, dtype=np.int8)
    terrain = Terrain(
        width=256, height=256, max_elevation=100.0,
        heightmap=np.zeros((256, 256), dtype=np.float32),
        biome_map=np.zeros((256, 256), dtype=np.int32),
    )
    return WorldState(
        tick=tick, elapsed=elapsed,
        terrain=terrain,
        drones=drones,
        survivors=survivors,
        fog_grid=fog,
        events=events,
        evidence=evidence,
        search_map=search_map,
    )


def _drone(
    id_: int = 0, x: float = 0.0, z: float = 0.0,
    status: DroneStatus = DroneStatus.ACTIVE, battery: float = 100.0,
) -> Drone:
    return Drone(
        id=id_, position=Vec3(x, 50.0, z),
        status=status, battery=battery, sensor_active=True,
    )


def _survivor(id_: int, discovered: bool = False) -> Survivor:
    return Survivor(id=id_, position=Vec3(100.0, 0.0, 100.0), discovered=discovered)


def _evidence(id_: int, discovered: bool = False) -> Evidence:
    return Evidence(
        id=id_, position=Vec3(100.0, 0.0, 100.0),
        kind=EvidenceKind.SIGNAL_FIRE.value, discovered=discovered,
    )


# ---------------------------------------------------------------------------
# Survivor discovery timing
# ---------------------------------------------------------------------------


class TestSurvivorTimeline:
    def test_records_first_discovery_time(self) -> None:
        m = MetricsTracker()
        m.record_tick(_mk_world(elapsed=10.0, survivors=(_survivor(1, False),)))
        m.record_tick(_mk_world(elapsed=20.0, survivors=(_survivor(1, True),)))
        card = m.scorecard()
        assert card["survivors"]["time_to_first"] == 20.0
        assert card["survivors"]["found"] == 1

    def test_records_per_survivor_times(self) -> None:
        m = MetricsTracker()
        m.record_tick(_mk_world(
            elapsed=10.0,
            survivors=(_survivor(1, True), _survivor(2, False), _survivor(3, False)),
        ))
        m.record_tick(_mk_world(
            elapsed=30.0,
            survivors=(_survivor(1, True), _survivor(2, True), _survivor(3, False)),
        ))
        m.record_tick(_mk_world(
            elapsed=50.0,
            survivors=(_survivor(1, True), _survivor(2, True), _survivor(3, True)),
        ))
        card = m.scorecard()
        assert card["survivors"]["discovery_timeline"] == [10.0, 30.0, 50.0]
        # MTTD = mean of discovery times = (10 + 30 + 50) / 3 = 30.0
        assert card["survivors"]["mttd_seconds"] == 30.0
        assert card["survivors"]["time_to_last"] == 50.0

    def test_mttd_is_none_when_no_finds(self) -> None:
        m = MetricsTracker()
        m.record_tick(_mk_world(elapsed=10.0, survivors=(_survivor(1, False),)))
        card = m.scorecard()
        assert card["survivors"]["mttd_seconds"] is None
        assert card["survivors"]["time_to_first"] is None


# ---------------------------------------------------------------------------
# Survival window hit rate (mission-dependent)
# ---------------------------------------------------------------------------


class _StubMission:
    """Minimal stand-in for SearchMission for tests."""

    def __init__(
        self,
        name: str = "stub",
        seed: int = 42,
        survival_window_seconds: float = 60.0,
        evidence: list[Evidence] | None = None,
    ) -> None:
        self.name = name
        self.seed = seed
        self.survival_window_seconds = survival_window_seconds
        self.evidence = evidence if evidence is not None else []


class TestSurvivalWindow:
    def test_counts_survivors_inside_window(self) -> None:
        mission = _StubMission(survival_window_seconds=60.0)
        m = MetricsTracker(mission=mission)
        # 2 found at t=30, 1 found at t=90 (past the window)
        m.record_tick(_mk_world(
            elapsed=30.0,
            survivors=(_survivor(1, True), _survivor(2, True), _survivor(3, False)),
        ))
        m.record_tick(_mk_world(
            elapsed=90.0,
            survivors=(_survivor(1, True), _survivor(2, True), _survivor(3, True)),
        ))
        card = m.scorecard()
        assert card["survivors"]["found_in_survival_window"] == 2
        assert card["survivors"]["survival_window_pct"] == round(2 / 3 * 100, 1)

    def test_no_mission_leaves_window_fields_none(self) -> None:
        m = MetricsTracker(mission=None)
        m.record_tick(_mk_world(elapsed=30.0, survivors=(_survivor(1, True),)))
        card = m.scorecard()
        assert card["survivors"]["found_in_survival_window"] is None
        assert card["survivors"]["survival_window_pct"] is None


# ---------------------------------------------------------------------------
# Evidence timing — the Phase 3 headline
# ---------------------------------------------------------------------------


class TestEvidenceTiming:
    def test_records_first_evidence_time(self) -> None:
        mission = _StubMission(evidence=[_evidence(1), _evidence(2)])
        m = MetricsTracker(mission=mission)
        m.record_tick(_mk_world(elapsed=5.0, evidence=(_evidence(1, False), _evidence(2, False))))
        m.record_tick(_mk_world(elapsed=15.0, evidence=(_evidence(1, True), _evidence(2, False))))
        card = m.scorecard()
        assert card["evidence"]["time_to_first"] == 15.0
        assert card["evidence"]["discovered"] == 1
        assert card["evidence"]["planted"] == 2

    def test_evidence_to_survivor_latency_captures_first_find_after_clue(self) -> None:
        """The load-bearing Phase 3 number: how long after the first
        evidence discovery does the first survivor turn up?"""
        mission = _StubMission(evidence=[_evidence(1)])
        m = MetricsTracker(mission=mission)
        # t=10: evidence found, no survivors yet
        m.record_tick(_mk_world(
            elapsed=10.0,
            evidence=(_evidence(1, True),),
            survivors=(_survivor(1, False),),
        ))
        # t=35: survivor found
        m.record_tick(_mk_world(
            elapsed=35.0,
            evidence=(_evidence(1, True),),
            survivors=(_survivor(1, True),),
        ))
        card = m.scorecard()
        assert card["evidence"]["evidence_to_survivor_latency"] == 25.0

    def test_latency_ignores_survivors_found_before_evidence(self) -> None:
        """If a survivor was already found before the first evidence,
        that one doesn't count for the latency metric."""
        mission = _StubMission(evidence=[_evidence(1)])
        m = MetricsTracker(mission=mission)
        # t=5: survivor found (before any evidence)
        m.record_tick(_mk_world(
            elapsed=5.0,
            evidence=(_evidence(1, False),),
            survivors=(_survivor(1, True),),
        ))
        # t=20: evidence found
        m.record_tick(_mk_world(
            elapsed=20.0,
            evidence=(_evidence(1, True),),
            survivors=(_survivor(1, True),),
        ))
        # t=50: second survivor found (first real "after-evidence" find)
        m.record_tick(_mk_world(
            elapsed=50.0,
            evidence=(_evidence(1, True),),
            survivors=(_survivor(1, True), _survivor(2, True)),
        ))
        card = m.scorecard()
        # Latency = 50 - 20 = 30, not 5 - 20 (negative)
        assert card["evidence"]["evidence_to_survivor_latency"] == 30.0


# ---------------------------------------------------------------------------
# PoC entropy
# ---------------------------------------------------------------------------


class TestEntropy:
    def test_uniform_grid_has_max_entropy(self) -> None:
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        # Max entropy for 256*256 = 65536 cells is ln(65536) ≈ 11.09 nats
        ent = _shannon_entropy(sm.poc)
        expected = math.log(sm.poc.size)
        assert abs(ent - expected) < 0.01

    def test_spike_grid_has_low_entropy(self) -> None:
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.poc[100, 100] = 1.0
        ent = _shannon_entropy(sm.poc)
        # A single-cell spike has entropy 0 (all mass in one cell)
        assert ent < 0.01

    def test_entropy_drops_after_narrowing_updates(self) -> None:
        mission = _StubMission()
        m = MetricsTracker(mission=mission)
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)

        # Force metrics to sample immediately by spacing ticks past the
        # entropy sample interval (2s).
        m.record_tick(_mk_world(elapsed=0.0, search_map=sm))
        initial = m._initial_entropy
        assert initial is not None

        # Concentrate mass in a small region
        sm.poc[:, :] = np.float32(0.0)
        sm.add_gaussian((5000, 5000), radius_meters=200, weight=1.0)
        sm.normalize()

        m.record_tick(_mk_world(elapsed=3.0, search_map=sm))
        card = m.scorecard()
        assert card["search_quality"]["final_entropy"] < initial
        assert card["search_quality"]["entropy_drop_pct"] > 0


# ---------------------------------------------------------------------------
# Resource accounting
# ---------------------------------------------------------------------------


class TestResources:
    def test_accumulates_drone_distance(self) -> None:
        m = MetricsTracker()
        m.record_tick(_mk_world(drones=(_drone(0, x=0.0, z=0.0),)))
        m.record_tick(_mk_world(drones=(_drone(0, x=30.0, z=40.0),)))
        # Distance = √(30² + 40²) = 50m
        card = m.scorecard()
        assert abs(card["resources"]["total_drone_km"] - 0.05) < 1e-3

    def test_counts_active_ticks_vs_returning(self) -> None:
        m = MetricsTracker()
        m.record_tick(_mk_world(drones=(_drone(0, status=DroneStatus.ACTIVE),)))
        m.record_tick(_mk_world(drones=(_drone(0, status=DroneStatus.ACTIVE),)))
        m.record_tick(_mk_world(drones=(_drone(0, status=DroneStatus.RETURNING),)))
        card = m.scorecard()
        assert card["resources"]["drone_ticks_active"] == 2
        assert card["resources"]["drone_ticks_returning"] == 1
        # active_fraction = 2 / 3
        assert abs(card["resources"]["active_fraction"] - 2 / 3) < 1e-3

    def test_meters_per_find(self) -> None:
        m = MetricsTracker()
        m.record_tick(_mk_world(
            drones=(_drone(0, x=0.0, z=0.0),),
            survivors=(_survivor(1, False),),
        ))
        m.record_tick(_mk_world(
            elapsed=1.0,
            drones=(_drone(0, x=100.0, z=0.0),),
            survivors=(_survivor(1, True),),
        ))
        card = m.scorecard()
        assert card["resources"]["meters_per_find"] == 100.0

    def test_skips_teleport_sized_steps(self) -> None:
        """A world reset can snap a drone's position by kilometers; we
        shouldn't credit that as flight distance."""
        m = MetricsTracker()
        m.record_tick(_mk_world(drones=(_drone(0, x=0.0, z=0.0),)))
        m.record_tick(_mk_world(drones=(_drone(0, x=5000.0, z=5000.0),)))
        card = m.scorecard()
        # Guard clause rejects steps > 200m, so distance stays ~0
        assert card["resources"]["total_drone_km"] < 0.001


# ---------------------------------------------------------------------------
# Scorecard shape & serialization
# ---------------------------------------------------------------------------


class TestScorecardShape:
    def test_scorecard_has_expected_sections(self) -> None:
        m = MetricsTracker(mission=_StubMission())
        m.record_tick(_mk_world())
        card = m.scorecard()
        assert set(card.keys()) >= {
            "mission", "seed", "elapsed",
            "survivors", "evidence", "search_quality",
            "resources", "events",
        }

    def test_scorecard_survives_empty_world(self) -> None:
        """A tracker that's never been fed must still produce a scorecard
        without raising. Degenerate values (zeros, Nones) are fine."""
        m = MetricsTracker()
        card = m.scorecard()
        assert card["survivors"]["found"] == 0
        assert card["survivors"]["mttd_seconds"] is None

    def test_scorecard_is_json_safe(self) -> None:
        import json
        m = MetricsTracker(mission=_StubMission(evidence=[_evidence(1)]))
        m.record_tick(_mk_world(
            elapsed=5.0,
            evidence=(_evidence(1, True),),
            survivors=(_survivor(1, True),),
        ))
        # Round-trip through JSON — this catches any np.float32/int64 leakage
        encoded = json.dumps(m.scorecard())
        decoded = json.loads(encoded)
        assert decoded["mission"] == "stub"


class TestSerializeCompactness:
    def test_serialize_has_live_hud_fields(self) -> None:
        mission = _StubMission()
        m = MetricsTracker(mission=mission)
        m.record_tick(_mk_world())
        payload = m.serialize()
        assert "latest" in payload
        assert "search_quality" in payload
        assert "evidence_progress" in payload
        assert "resources" in payload
        # No time series (those are behind get_summary())
        assert "entropy_series" not in payload
        assert "coverage_series" not in payload
