"""Performance metrics tracking for the drone swarm simulation.

Two levels of output:
  * `serialize()` — compact per-tick payload for the live HUD. Meant to
    be small enough to ship in every state_update (a few dozen numbers,
    no time series).
  * `scorecard()` — end-of-run structured summary, suitable for JSON
    dumps and run-to-run comparison. This is the artifact a human (or
    CI) reads to decide "is run B better than run A?".

Design principle: coverage % is no longer the primary success signal.
With Bayesian search + evidence (Phase 3+), the swarm *should not*
cover uniformly — it should concentrate on the posterior. So the
metrics here are search-quality-oriented:

  * mean time to discovery (MTTD) across *all* survivors
  * fraction of survivors found within the mission's survival window
  * PoC entropy trajectory (should drop as the posterior narrows)
  * time-to-first-evidence and evidence→survivor latency
  * resource efficiency (km flown, battery per find, idle fraction)
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.simulation.engine import get_coverage_pct
from src.simulation.types import DroneStatus, WorldState

# Maximum number of samples retained per time series.
_MAX_SAMPLES = 500

# Minimum interval (in sim-seconds) between time-series samples.
_SAMPLE_INTERVAL = 1.0

# Coverage threshold (%) that counts as "full coverage".
_FULL_COVERAGE_THRESHOLD = 99.0

# PoC entropy: only sample every N seconds (computing Shannon entropy on a
# 256x256 grid is a few ms, but no reason to do it every tick).
_ENTROPY_SAMPLE_INTERVAL = 2.0


class MetricsTracker:
    """Collects simulation metrics and produces live HUD data + end-of-run scorecards.

    Pass the active `SearchMission` at construction so the tracker can
    evaluate mission-relative signals (survival window, total planted
    evidence, etc.). Passing `None` is fine for tests; mission-dependent
    fields will just be `None` in the scorecard.
    """

    def __init__(self, mission: object | None = None) -> None:
        # Mission context (optional — scorecard degrades gracefully).
        # Not type-annotated as SearchMission to avoid a circular import.
        self._mission = mission

        # --- Existing time series ---
        self._coverage_series: list[tuple[float, float]] = []
        self._survivors_series: list[tuple[float, int]] = []
        self._active_drones_series: list[tuple[float, int]] = []
        self._avg_battery_series: list[tuple[float, float]] = []

        # --- New time series ---
        # PoC Shannon entropy — drops as the posterior narrows. Good search
        # behavior shows a steady decline here.
        self._entropy_series: list[tuple[float, float]] = []
        self._last_entropy_sample_elapsed: float = -_ENTROPY_SAMPLE_INTERVAL

        # --- Milestone metrics ---
        self._time_to_first_discovery: float | None = None
        self._time_to_full_coverage: float | None = None
        self._time_to_first_evidence: float | None = None
        # Time between first evidence and the next survivor found after it —
        # the load-bearing number for the Phase 3 claim "evidence helps".
        # None until we see both in sequence.
        self._evidence_to_survivor_latency: float | None = None

        # --- Per-entity discovery times (id → elapsed-seconds-at-discovery) ---
        self._survivor_discovery_times: dict[int, float] = {}
        self._evidence_discovery_times: dict[int, float] = {}

        # --- Event counts by type name ---
        self._event_counts: dict[str, int] = defaultdict(int)

        # --- Resource accounting ---
        # Drone positions at the previous tick, so we can accumulate distance.
        self._last_drone_xz: dict[int, tuple[float, float]] = {}
        self._total_drone_distance_m: float = 0.0
        # Drone-tick buckets: how many (drone × tick) slots were spent in
        # each status. Fraction-active = active / total.
        self._drone_ticks_active: int = 0
        self._drone_ticks_returning: int = 0
        self._drone_ticks_recharging: int = 0
        self._drone_ticks_failed: int = 0

        # --- Tracking helpers ---
        self._last_sample_elapsed: float = -_SAMPLE_INTERVAL  # force first sample
        self._total_survivors_seen: int = 0
        self._latest_coverage: float = 0.0
        self._latest_entropy: float | None = None
        self._latest_elapsed: float = 0.0
        self._latest_survivors_found: int = 0
        self._latest_active_drones: int = 0
        self._latest_avg_battery: float = 100.0
        self._initial_entropy: float | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tick(self, world: WorldState) -> None:
        """Record metrics from the current tick."""
        elapsed = world.elapsed
        self._latest_elapsed = elapsed

        # --- Scalar snapshots ---
        coverage_pct = get_coverage_pct(world.fog_grid)
        self._latest_coverage = coverage_pct

        survivors_found = sum(1 for s in world.survivors if s.discovered)
        self._latest_survivors_found = survivors_found
        self._total_survivors_seen = max(
            self._total_survivors_seen, len(world.survivors),
        )

        active_drones = sum(
            1 for d in world.drones if d.status == DroneStatus.ACTIVE
        )
        self._latest_active_drones = active_drones

        avg_battery = (
            sum(d.battery for d in world.drones) / len(world.drones)
            if world.drones else 0.0
        )
        self._latest_avg_battery = avg_battery

        # --- Per-survivor discovery times ---
        # Record the first tick each survivor was marked discovered. Use
        # the survivor's own discovered_at_tick × tick_rate if set, else
        # current elapsed.
        for s in world.survivors:
            if s.discovered and s.id not in self._survivor_discovery_times:
                self._survivor_discovery_times[s.id] = elapsed

        # --- Per-evidence discovery times ---
        for e in world.evidence:
            if e.discovered and e.id not in self._evidence_discovery_times:
                self._evidence_discovery_times[e.id] = elapsed
                if self._time_to_first_evidence is None:
                    self._time_to_first_evidence = elapsed

        # --- Milestones ---
        if self._time_to_first_discovery is None and survivors_found > 0:
            self._time_to_first_discovery = elapsed

        if (
            self._time_to_full_coverage is None
            and coverage_pct >= _FULL_COVERAGE_THRESHOLD
        ):
            self._time_to_full_coverage = elapsed

        # First survivor found strictly after first evidence → the "evidence
        # helped" signal. Captured once, never overwritten.
        if (
            self._evidence_to_survivor_latency is None
            and self._time_to_first_evidence is not None
        ):
            for t in self._survivor_discovery_times.values():
                if t > self._time_to_first_evidence:
                    self._evidence_to_survivor_latency = (
                        t - self._time_to_first_evidence
                    )
                    break

        # --- Event tallies ---
        for event in world.events:
            self._event_counts[event.type.name.lower()] += 1

        # --- Drone bucket accounting ---
        # Note: a "drone-tick" is one drone × one tick. Summing these lets
        # us express utilization as a fraction (active ticks / all ticks).
        for d in world.drones:
            if d.status == DroneStatus.ACTIVE:
                self._drone_ticks_active += 1
            elif d.status == DroneStatus.RETURNING:
                self._drone_ticks_returning += 1
            elif d.status == DroneStatus.RECHARGING:
                self._drone_ticks_recharging += 1
            elif d.status == DroneStatus.FAILED:
                self._drone_ticks_failed += 1

            # Total distance flown (XZ only; altitude changes don't count
            # for mission coverage).
            prev = self._last_drone_xz.get(d.id)
            if prev is not None and d.status != DroneStatus.FAILED:
                dx = d.position.x - prev[0]
                dz = d.position.z - prev[1]
                step = math.sqrt(dx * dx + dz * dz)
                # Guard against teleport-on-reset; anything >200m in one tick
                # at 10Hz would be >2km/s, clearly not a real flight step.
                if step < 200.0:
                    self._total_drone_distance_m += step
            self._last_drone_xz[d.id] = (d.position.x, d.position.z)

        # --- Time-series sampling ---
        if elapsed - self._last_sample_elapsed >= _SAMPLE_INTERVAL:
            self._last_sample_elapsed = elapsed
            self._append_sample(self._coverage_series, elapsed, round(coverage_pct, 1))
            self._append_sample(self._survivors_series, elapsed, survivors_found)
            self._append_sample(self._active_drones_series, elapsed, active_drones)
            self._append_sample(self._avg_battery_series, elapsed, round(avg_battery, 1))

        # --- PoC entropy ---
        # Separate (slower) cadence; computing Shannon entropy touches the
        # full grid. Capture the first sample as the baseline so we can
        # report "entropy dropped by X%" in the scorecard.
        if (
            world.search_map is not None
            and elapsed - self._last_entropy_sample_elapsed
            >= _ENTROPY_SAMPLE_INTERVAL
        ):
            self._last_entropy_sample_elapsed = elapsed
            ent = _shannon_entropy(world.search_map.poc)
            self._latest_entropy = ent
            if self._initial_entropy is None:
                self._initial_entropy = ent
            self._append_sample(self._entropy_series, elapsed, round(ent, 4))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a complete metrics summary for display."""
        return {
            "coverage_series": self._coverage_series,
            "survivors_series": self._survivors_series,
            "active_drones_series": self._active_drones_series,
            "avg_battery_series": self._avg_battery_series,
            "entropy_series": self._entropy_series,
            "time_to_first_discovery": self._time_to_first_discovery,
            "time_to_full_coverage": self._time_to_full_coverage,
            "time_to_first_evidence": self._time_to_first_evidence,
            "evidence_to_survivor_latency": self._evidence_to_survivor_latency,
            "event_counts": dict(self._event_counts),
            "efficiency_score": self._efficiency_score(),
            "coverage_rate": self._coverage_rate(),
            "latest": self._latest_snapshot(),
        }

    def serialize(self) -> dict:
        """Lightweight per-tick payload for live HUD broadcast.

        Keeps the message small: latest scalars + a tight summary of the
        scorecard's most informative fields (MTTD, entropy drop, evidence
        progress). Full time series live behind `get_summary()`.
        """
        return {
            "latest": self._latest_snapshot(),
            "time_to_first_discovery": self._time_to_first_discovery,
            "time_to_full_coverage": self._time_to_full_coverage,
            "time_to_first_evidence": self._time_to_first_evidence,
            "evidence_to_survivor_latency": self._evidence_to_survivor_latency,
            "event_counts": dict(self._event_counts),
            "efficiency_score": self._efficiency_score(),
            "coverage_rate": self._coverage_rate(),
            "search_quality": self._search_quality_summary(),
            "evidence_progress": self._evidence_progress_summary(),
            "resources": self._resource_summary(),
        }

    def scorecard(self) -> dict:
        """Structured end-of-run summary. Suitable for JSON dumps.

        This is the comparison artifact — run a mission, dump the
        scorecard, commit it alongside the change, diff later.
        """
        mission_name = getattr(self._mission, "name", None)
        mission_seed = getattr(self._mission, "seed", None)
        survival_window = getattr(self._mission, "survival_window_seconds", None)
        planted_evidence_count = (
            len(self._mission.evidence) if self._mission is not None else 0
        )

        discovery_times_sorted = sorted(self._survivor_discovery_times.values())
        mttd = (
            sum(discovery_times_sorted) / len(discovery_times_sorted)
            if discovery_times_sorted else None
        )
        time_to_last = discovery_times_sorted[-1] if discovery_times_sorted else None

        found_in_window = None
        survival_window_pct = None
        if survival_window is not None:
            in_window = sum(1 for t in discovery_times_sorted if t <= survival_window)
            found_in_window = in_window
            total = self._total_survivors_seen or len(discovery_times_sorted)
            survival_window_pct = round(
                (in_window / total) * 100.0, 1,
            ) if total > 0 else None

        total_drone_ticks = (
            self._drone_ticks_active
            + self._drone_ticks_returning
            + self._drone_ticks_recharging
            + self._drone_ticks_failed
        )
        active_fraction = (
            self._drone_ticks_active / total_drone_ticks
            if total_drone_ticks > 0 else None
        )

        battery_per_find = None
        if self._latest_survivors_found > 0:
            # Rough proxy: total drone-km divided by finds, then scaled by
            # a drone's typical battery-per-km. Simpler: km / finds is the
            # most useful comparison statistic, keep it honest.
            battery_per_find = round(
                self._total_drone_distance_m / self._latest_survivors_found, 1,
            )

        entropy_drop_pct = None
        if self._initial_entropy is not None and self._latest_entropy is not None:
            if self._initial_entropy > 1e-9:
                entropy_drop_pct = round(
                    (1.0 - self._latest_entropy / self._initial_entropy) * 100.0, 1,
                )

        return {
            "mission": mission_name,
            "seed": mission_seed,
            "elapsed": round(self._latest_elapsed, 2),
            "survivors": {
                "found": self._latest_survivors_found,
                "total_placed": self._total_survivors_seen,
                "found_in_survival_window": found_in_window,
                "survival_window_pct": survival_window_pct,
                "mttd_seconds": round(mttd, 2) if mttd is not None else None,
                "time_to_first": (
                    round(self._time_to_first_discovery, 2)
                    if self._time_to_first_discovery is not None else None
                ),
                "time_to_last": (
                    round(time_to_last, 2) if time_to_last is not None else None
                ),
                "discovery_timeline": [
                    round(t, 2) for t in discovery_times_sorted
                ],
            },
            "evidence": {
                "discovered": len(self._evidence_discovery_times),
                "planted": planted_evidence_count,
                "time_to_first": (
                    round(self._time_to_first_evidence, 2)
                    if self._time_to_first_evidence is not None else None
                ),
                "evidence_to_survivor_latency": (
                    round(self._evidence_to_survivor_latency, 2)
                    if self._evidence_to_survivor_latency is not None else None
                ),
            },
            "search_quality": {
                "initial_entropy": (
                    round(self._initial_entropy, 4)
                    if self._initial_entropy is not None else None
                ),
                "final_entropy": (
                    round(self._latest_entropy, 4)
                    if self._latest_entropy is not None else None
                ),
                "entropy_drop_pct": entropy_drop_pct,
                "coverage_pct": round(self._latest_coverage, 1),
            },
            "resources": {
                "total_drone_km": round(self._total_drone_distance_m / 1000.0, 3),
                "meters_per_find": battery_per_find,
                "active_fraction": (
                    round(active_fraction, 3)
                    if active_fraction is not None else None
                ),
                "drone_ticks_active": self._drone_ticks_active,
                "drone_ticks_returning": self._drone_ticks_returning,
                "drone_ticks_recharging": self._drone_ticks_recharging,
                "drone_ticks_failed": self._drone_ticks_failed,
            },
            "events": dict(self._event_counts),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _latest_snapshot(self) -> dict:
        return {
            "elapsed": round(self._latest_elapsed, 2),
            "coverage_pct": round(self._latest_coverage, 1),
            "survivors_found": self._latest_survivors_found,
            "total_survivors": self._total_survivors_seen,
            "active_drones": self._latest_active_drones,
            "avg_battery": round(self._latest_avg_battery, 1),
            "entropy": (
                round(self._latest_entropy, 4)
                if self._latest_entropy is not None else None
            ),
        }

    def _efficiency_score(self) -> float | None:
        if self._latest_elapsed <= 0:
            return None
        return round(self._latest_survivors_found / self._latest_elapsed * 100.0, 2)

    def _coverage_rate(self) -> float | None:
        if self._latest_elapsed <= 0:
            return None
        return round(self._latest_coverage / self._latest_elapsed, 3)

    def _search_quality_summary(self) -> dict:
        """Tight "is this a good run?" snapshot for the HUD."""
        discovery_times = sorted(self._survivor_discovery_times.values())
        mttd = (
            sum(discovery_times) / len(discovery_times)
            if discovery_times else None
        )
        survival_window = getattr(self._mission, "survival_window_seconds", None)
        survival_window_pct = None
        if survival_window is not None and self._total_survivors_seen > 0:
            in_window = sum(1 for t in discovery_times if t <= survival_window)
            survival_window_pct = round(
                (in_window / self._total_survivors_seen) * 100.0, 1,
            )

        entropy_drop_pct = None
        if (
            self._initial_entropy is not None
            and self._latest_entropy is not None
            and self._initial_entropy > 1e-9
        ):
            entropy_drop_pct = round(
                (1.0 - self._latest_entropy / self._initial_entropy) * 100.0, 1,
            )

        return {
            "mttd_seconds": round(mttd, 2) if mttd is not None else None,
            "survival_window_pct": survival_window_pct,
            "entropy_drop_pct": entropy_drop_pct,
        }

    def _evidence_progress_summary(self) -> dict:
        planted = (
            len(self._mission.evidence) if self._mission is not None else 0
        )
        return {
            "discovered": len(self._evidence_discovery_times),
            "planted": planted,
        }

    def _resource_summary(self) -> dict:
        total = (
            self._drone_ticks_active
            + self._drone_ticks_returning
            + self._drone_ticks_recharging
            + self._drone_ticks_failed
        )
        active_fraction = (
            round(self._drone_ticks_active / total, 3) if total > 0 else None
        )
        return {
            "total_drone_km": round(self._total_drone_distance_m / 1000.0, 3),
            "active_fraction": active_fraction,
        }

    @staticmethod
    def _append_sample(series: list, elapsed: float, value: object) -> None:
        series.append((round(elapsed, 2), value))
        if len(series) > _MAX_SAMPLES:
            del series[0]


def _shannon_entropy(poc: np.ndarray) -> float:
    """Shannon entropy of the PoC distribution (in nats).

    Treats the grid as a discrete probability distribution. A fully
    uniform grid has maximum entropy; a sharp spike has minimum. Good
    search behavior drops this over time.
    """
    flat = poc.ravel().astype(np.float64)
    total = float(flat.sum())
    if total <= 1e-12:
        return 0.0
    p = flat / total
    # Avoid log(0) — mask to positive cells.
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())
