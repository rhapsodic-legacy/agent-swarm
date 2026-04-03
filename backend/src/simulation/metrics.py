"""Performance metrics tracking for the drone swarm simulation.

Records time-series data and summary statistics from each simulation tick.
Designed for dashboard display and WebSocket broadcast.
"""

from __future__ import annotations

from collections import defaultdict

from src.simulation.engine import get_coverage_pct
from src.simulation.types import DroneStatus, WorldState

# Maximum number of samples retained per time series.
_MAX_SAMPLES = 500

# Minimum interval (in sim-seconds) between time-series samples.
_SAMPLE_INTERVAL = 1.0

# Coverage threshold (%) that counts as "full coverage".
_FULL_COVERAGE_THRESHOLD = 99.0


class MetricsTracker:
    """Collects and summarises simulation performance metrics.

    Call ``record_tick`` once per simulation tick. Query results with
    ``get_summary`` (full snapshot) or ``serialize`` (lightweight payload
    suitable for WebSocket broadcast).
    """

    def __init__(self) -> None:
        # Time-series data: list of (elapsed, value) tuples.
        self._coverage_series: list[tuple[float, float]] = []
        self._survivors_series: list[tuple[float, int]] = []
        self._active_drones_series: list[tuple[float, int]] = []
        self._avg_battery_series: list[tuple[float, float]] = []

        # Milestone metrics.
        self._time_to_first_discovery: float | None = None
        self._time_to_full_coverage: float | None = None

        # Event counts by type name.
        self._event_counts: dict[str, int] = defaultdict(int)

        # Tracking helpers.
        self._last_sample_elapsed: float = -_SAMPLE_INTERVAL  # force first sample
        self._total_survivors: int = 0
        self._latest_coverage: float = 0.0
        self._latest_elapsed: float = 0.0
        self._latest_survivors_found: int = 0
        self._latest_active_drones: int = 0
        self._latest_avg_battery: float = 100.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tick(self, world: WorldState) -> None:
        """Record metrics from the current tick."""
        elapsed = world.elapsed
        self._latest_elapsed = elapsed

        # --- Compute current values ---
        coverage_pct = get_coverage_pct(world.fog_grid)
        self._latest_coverage = coverage_pct

        survivors_found = sum(1 for s in world.survivors if s.discovered)
        self._latest_survivors_found = survivors_found
        self._total_survivors = len(world.survivors)

        active_drones = sum(1 for d in world.drones if d.status == DroneStatus.ACTIVE)
        self._latest_active_drones = active_drones

        avg_battery = (
            sum(d.battery for d in world.drones) / len(world.drones) if world.drones else 0.0
        )
        self._latest_avg_battery = avg_battery

        # --- Milestone checks ---
        if self._time_to_first_discovery is None and survivors_found > 0:
            self._time_to_first_discovery = elapsed

        if self._time_to_full_coverage is None and coverage_pct >= _FULL_COVERAGE_THRESHOLD:
            self._time_to_full_coverage = elapsed

        # --- Accumulate event counts ---
        for event in world.events:
            self._event_counts[event.type.name.lower()] += 1

        # --- Sample time series (at most once per _SAMPLE_INTERVAL sim-seconds) ---
        if elapsed - self._last_sample_elapsed >= _SAMPLE_INTERVAL:
            self._last_sample_elapsed = elapsed
            self._append_sample(self._coverage_series, elapsed, round(coverage_pct, 1))
            self._append_sample(self._survivors_series, elapsed, survivors_found)
            self._append_sample(self._active_drones_series, elapsed, active_drones)
            self._append_sample(self._avg_battery_series, elapsed, round(avg_battery, 1))

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
            "time_to_first_discovery": self._time_to_first_discovery,
            "time_to_full_coverage": self._time_to_full_coverage,
            "event_counts": dict(self._event_counts),
            "efficiency_score": self._efficiency_score(),
            "coverage_rate": self._coverage_rate(),
            "latest": self._latest_snapshot(),
        }

    def serialize(self) -> dict:
        """Serialize a lightweight payload for WebSocket broadcast.

        Returns only the latest data point and summary statistics (not the
        full time series, which would be too large for every broadcast).
        """
        return {
            "latest": self._latest_snapshot(),
            "time_to_first_discovery": self._time_to_first_discovery,
            "time_to_full_coverage": self._time_to_full_coverage,
            "event_counts": dict(self._event_counts),
            "efficiency_score": self._efficiency_score(),
            "coverage_rate": self._coverage_rate(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _latest_snapshot(self) -> dict:
        """Current-tick values as a flat dict."""
        return {
            "elapsed": round(self._latest_elapsed, 2),
            "coverage_pct": round(self._latest_coverage, 1),
            "survivors_found": self._latest_survivors_found,
            "total_survivors": self._total_survivors,
            "active_drones": self._latest_active_drones,
            "avg_battery": round(self._latest_avg_battery, 1),
        }

    def _efficiency_score(self) -> float | None:
        """survivors_found / elapsed_seconds * 100. None if no time has passed."""
        if self._latest_elapsed <= 0:
            return None
        return round(self._latest_survivors_found / self._latest_elapsed * 100.0, 2)

    def _coverage_rate(self) -> float | None:
        """coverage_pct / elapsed_seconds. None if no time has passed."""
        if self._latest_elapsed <= 0:
            return None
        return round(self._latest_coverage / self._latest_elapsed, 3)

    @staticmethod
    def _append_sample(series: list, elapsed: float, value: object) -> None:
        """Append a sample, enforcing the max-samples cap."""
        series.append((round(elapsed, 2), value))
        if len(series) > _MAX_SAMPLES:
            # Drop the oldest sample.
            del series[0]
