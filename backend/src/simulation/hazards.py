"""Terrain hazards: no-fly zones and signal jammers.

Hazards are static obstacles generated at simulation init. They affect
drone behavior — no-fly zones should be avoided by pathfinding, and
signal jammers disable communications for drones within their radius.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from src.simulation.types import Drone, Vec3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hazard:
    """A terrain hazard."""

    id: int
    type: str  # "no_fly_zone" or "signal_jammer"
    center: Vec3  # center position (XZ plane)
    radius: float  # effect radius in meters


class HazardSystem:
    """Generates and manages terrain hazards."""

    def __init__(self, terrain_width: int, terrain_height: int, seed: int, count: int = 4):
        """Generate random hazards on the terrain.

        Args:
            terrain_width: Terrain width in meters.
            terrain_height: Terrain height in meters.
            seed: RNG seed for reproducibility.
            count: Total number of hazards to generate.
        """
        rng = np.random.default_rng(seed)
        margin = 20
        hazards: list[Hazard] = []

        for i in range(count):
            # First half are no-fly zones, second half are signal jammers
            hazard_type = "no_fly_zone" if i < count // 2 else "signal_jammer"

            x = float(rng.uniform(margin, terrain_width - margin))
            z = float(rng.uniform(margin, terrain_height - margin))
            center = Vec3(x, 0.0, z)

            if hazard_type == "no_fly_zone":
                radius = float(rng.uniform(15.0, 30.0))
            else:
                radius = float(rng.uniform(20.0, 40.0))

            hazards.append(Hazard(id=i, type=hazard_type, center=center, radius=radius))

        self._hazards = tuple(hazards)

    def get_hazards(self) -> tuple[Hazard, ...]:
        """Return all hazards."""
        return self._hazards

    def is_in_no_fly_zone(self, position: Vec3) -> bool:
        """Check if a position is inside any no-fly zone."""
        for h in self._hazards:
            if h.type != "no_fly_zone":
                continue
            dx = position.x - h.center.x
            dz = position.z - h.center.z
            dist = (dx**2 + dz**2) ** 0.5
            if dist <= h.radius:
                return True
        return False

    def is_jammed(self, position: Vec3) -> bool:
        """Check if a position is affected by a signal jammer."""
        for h in self._hazards:
            if h.type != "signal_jammer":
                continue
            dx = position.x - h.center.x
            dz = position.z - h.center.z
            dist = (dx**2 + dz**2) ** 0.5
            if dist <= h.radius:
                return True
        return False

    def apply_to_drone(self, drone: Drone) -> Drone:
        """Apply hazard effects to a drone. Returns modified drone.

        - No-fly zone: logs a warning (pathfinding should avoid it).
        - Signal jammer: sets comms_active = False while inside, restores it outside.
        """
        in_no_fly = self.is_in_no_fly_zone(drone.position)
        if in_no_fly:
            logger.debug("Drone %d is inside a no-fly zone at %s", drone.id, drone.position)

        jammed = self.is_jammed(drone.position)
        if jammed and drone.comms_active:
            return replace(drone, comms_active=False)
        if not jammed and not drone.comms_active:
            return replace(drone, comms_active=True)

        return drone

    def serialize(self) -> list[dict]:
        """Serialize hazards for WebSocket broadcast."""
        return [
            {
                "id": h.id,
                "type": h.type,
                "center": [h.center.x, h.center.y, h.center.z],
                "radius": h.radius,
            }
            for h in self._hazards
        ]
