"""Replay recording and playback for the drone swarm simulation.

Records periodic snapshots of the simulation state so that a completed (or
in-progress) run can be replayed in the frontend without re-simulating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.simulation.engine import get_coverage_pct
from src.simulation.types import WorldState


@dataclass
class ReplayFrame:
    """A single recorded snapshot of the simulation."""

    tick: int
    elapsed: float
    drones: list[dict]
    survivors: list[dict]
    coverage_pct: float
    events: list[dict]


class ReplayRecorder:
    """Records simulation frames at a configurable interval.

    Args:
        record_interval: Record every *n*-th tick.  With a 20 Hz tick rate
            and ``record_interval=5`` this yields ~4 frames per second,
            keeping file sizes manageable.
    """

    def __init__(self, record_interval: int = 5) -> None:
        self._record_interval: int = record_interval
        self._frames: list[ReplayFrame] = []
        self._recording: bool = False
        # Header metadata — populated from the first recorded world state.
        self._terrain_seed: int | None = None
        self._drone_count: int | None = None
        self._terrain_size: int | None = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin (or resume) recording."""
        self._recording = True

    def stop(self) -> None:
        """Pause recording."""
        self._recording = False

    def is_recording(self) -> bool:
        return self._recording

    def get_frame_count(self) -> int:
        return len(self._frames)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, world: WorldState) -> None:
        """Record a frame if recording is active and this is a sampled tick."""
        if not self._recording:
            return

        # Populate header metadata on first record.
        if self._terrain_seed is None:
            self._terrain_seed = world.terrain.seed
            self._drone_count = len(world.drones)
            self._terrain_size = world.terrain.width

        if world.tick % self._record_interval != 0:
            return

        frame = ReplayFrame(
            tick=world.tick,
            elapsed=_r1(world.elapsed),
            drones=[_serialize_drone(d) for d in world.drones],
            survivors=[_serialize_survivor(s) for s in world.survivors if s.discovered],
            coverage_pct=_r1(get_coverage_pct(world.fog_grid)),
            events=[_serialize_event(e) for e in world.events],
        )
        self._frames.append(frame)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the recorded replay to a JSON file."""
        duration = self._frames[-1].elapsed if self._frames else 0.0

        data = {
            "header": {
                "terrain_seed": self._terrain_seed,
                "drone_count": self._drone_count,
                "terrain_size": self._terrain_size,
                "total_frames": len(self._frames),
                "duration": _r1(duration),
            },
            "frames": [_frame_to_dict(f) for f in self._frames],
        }

        with open(path, "w") as fh:
            json.dump(data, fh, separators=(",", ":"))


class ReplayPlayer:
    """Provides random-access playback of a previously recorded replay."""

    def __init__(self) -> None:
        self._frames: list[ReplayFrame] = []
        self._header: dict = {}

    def load(self, path: str) -> None:
        """Load a replay from a JSON file written by ``ReplayRecorder.save``."""
        with open(path) as fh:
            data = json.load(fh)

        self._header = data.get("header", {})
        self._frames = [_dict_to_frame(d) for d in data.get("frames", [])]

    def get_frame(self, index: int) -> ReplayFrame | None:
        """Return the frame at *index*, or ``None`` if out of range."""
        if 0 <= index < len(self._frames):
            return self._frames[index]
        return None

    def get_frame_count(self) -> int:
        return len(self._frames)

    def get_duration(self) -> float:
        """Total elapsed sim-time of the replay."""
        if self._frames:
            return self._frames[-1].elapsed
        return 0.0


# ======================================================================
# Private helpers
# ======================================================================


def _r1(value: float) -> float:
    """Round a float to 1 decimal place."""
    return round(value, 1)


def _serialize_drone(drone: object) -> dict:
    """Serialize a Drone to a compact dict."""
    # Import locally to keep type-only dependency lightweight.
    from src.simulation.types import Drone

    d: Drone = drone  # type: ignore[assignment]
    return {
        "id": d.id,
        "position": [_r1(d.position.x), _r1(d.position.y), _r1(d.position.z)],
        "battery": _r1(d.battery),
        "status": d.status.name.lower(),
    }


def _serialize_survivor(survivor: object) -> dict:
    from src.simulation.types import Survivor

    s: Survivor = survivor  # type: ignore[assignment]
    return {
        "id": s.id,
        "position": [_r1(s.position.x), _r1(s.position.y), _r1(s.position.z)],
    }


def _serialize_event(event: object) -> dict:
    from src.simulation.types import SimEvent

    e: SimEvent = event  # type: ignore[assignment]
    return {
        "type": e.type.name.lower(),
        "tick": e.tick,
        "drone_id": e.drone_id,
        "survivor_id": e.survivor_id,
    }


def _frame_to_dict(frame: ReplayFrame) -> dict:
    return {
        "tick": frame.tick,
        "elapsed": frame.elapsed,
        "drones": frame.drones,
        "survivors": frame.survivors,
        "coverage_pct": frame.coverage_pct,
        "events": frame.events,
    }


def _dict_to_frame(d: dict) -> ReplayFrame:
    return ReplayFrame(
        tick=d["tick"],
        elapsed=d["elapsed"],
        drones=d["drones"],
        survivors=d["survivors"],
        coverage_pct=d["coverage_pct"],
        events=d["events"],
    )
