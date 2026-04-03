"""Dynamic wind system for the drone swarm simulation.

Provides global wind that shifts direction and intensity over time,
plus local spatial variation. Wind vectors are added to drone velocity
computation externally.
"""

from __future__ import annotations

import math

from src.simulation.types import Vec3


class WeatherSystem:
    """Simulates dynamic wind with smooth temporal and spatial variation.

    Global wind direction and speed evolve over time using layered sine
    waves (cheap Perlin-like smoothness, no external deps). Local wind
    at each (x, z) position is perturbed via a deterministic spatial hash
    so nearby drones experience slightly different conditions.
    """

    def __init__(self, seed: int, terrain_size: int) -> None:
        self._seed = seed
        self._terrain_size = terrain_size

        # Derive deterministic base parameters from seed
        rng = seed % 1_000_000
        self._base_speed: float = 3.0 + (rng % 500) / 100.0  # 3.0 – 7.99 m/s
        self._base_direction: float = (rng % 628) / 100.0  # 0 – ~2*pi radians

        # Current computed values (updated each frame)
        self._wind_speed: float = self._base_speed
        self._wind_direction: float = self._base_direction
        self._gusting: bool = False
        self._elapsed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, elapsed: float) -> None:
        """Advance weather state to the given elapsed time."""
        self._elapsed = elapsed

        # Direction drifts slowly
        self._wind_direction = self._base_direction + 0.3 * math.sin(elapsed * 0.05)

        # Speed oscillates around base
        speed_factor = 1.0 + 0.4 * math.sin(elapsed * 0.08)
        self._wind_speed = self._base_speed * speed_factor

        # Gusting: a secondary higher-frequency oscillation pushes speed over 80% of max
        gust_wave = math.sin(elapsed * 0.37) + 0.5 * math.sin(elapsed * 0.53)
        max_speed = self._base_speed * 1.4
        self._gusting = (self._wind_speed + gust_wave) > (max_speed * 0.8)

    def get_wind_at(self, x: float, z: float) -> Vec3:
        """Return the wind vector at world position (x, z).

        The global wind is perturbed slightly by a deterministic spatial
        hash so that each location experiences a unique but coherent breeze.
        The Y component is always 0 (horizontal wind only).
        """
        # Local perturbation from spatial hash (-0.15 .. +0.15 of base values)
        local = self._spatial_perturbation(x, z)

        direction = self._wind_direction + local * 0.15
        speed = self._wind_speed * (1.0 + local * 0.10)

        wx = math.cos(direction) * speed
        wz = math.sin(direction) * speed
        return Vec3(wx, 0.0, wz)

    def get_wind_speed(self) -> float:
        """Current global wind speed in m/s."""
        return self._wind_speed

    def get_wind_direction(self) -> float:
        """Current global wind direction in radians."""
        return self._wind_direction

    def serialize(self) -> dict:
        """Serialize current weather state for WebSocket broadcast."""
        return {
            "wind_speed": round(self._wind_speed, 3),
            "wind_direction": round(self._wind_direction, 4),
            "gusting": self._gusting,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spatial_perturbation(self, x: float, z: float) -> float:
        """Return a deterministic value in [-1, 1] that varies smoothly-ish
        across space, derived from quantised grid coords and the seed.

        Uses a simple sine-based hash so nearby points are correlated but
        not identical.
        """
        # Quantise to ~4 m cells for smooth-ish variation
        cell_x = math.floor(x / 4.0)
        cell_z = math.floor(z / 4.0)

        # Deterministic hash mixing
        h = (cell_x * 374761 + cell_z * 668265 + self._seed * 137) & 0xFFFFFFFF
        # Map to [-1, 1] via sine of a large prime multiple
        return math.sin(h * 0.00015473)
