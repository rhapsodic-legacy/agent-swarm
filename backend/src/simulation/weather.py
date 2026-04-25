"""Dynamic wind system for the drone swarm simulation.

Provides global wind that shifts direction and intensity over time,
plus local spatial variation. Wind vectors are added to drone velocity
computation externally.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from src.simulation.types import Vec3


@dataclass(frozen=True)
class GustRegion:
    """A discrete area where wind periodically gusts dangerously.

    Strength oscillates on its own sine cycle so different regions peak at
    different times — the operator sees a weather map that changes over the
    course of a mission, not a global on/off.
    """

    center_x: float
    center_z: float
    radius: float
    base_strength: float  # scales the oscillation amplitude (0..1)
    period: float  # seconds per full cycle
    phase: float  # phase offset, radians


class WeatherSystem:
    """Simulates dynamic wind with smooth temporal and spatial variation.

    Global wind direction and speed evolve over time using layered sine
    waves (cheap Perlin-like smoothness, no external deps). Local wind
    at each (x, z) position is perturbed via a deterministic spatial hash
    so nearby drones experience slightly different conditions.
    """

    # Threshold above which a gust region is considered hazardous (0..1).
    # Regions above this are filtered out of the priority market and rendered.
    GUST_HAZARD_THRESHOLD = 0.55

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

        # Discrete gust regions — seeded deterministically so the same seed
        # reproduces the same weather pattern.
        self._gust_regions: tuple[GustRegion, ...] = self._make_gust_regions(
            seed,
            terrain_size,
        )

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

    def gust_strength_at(self, x: float, z: float) -> float:
        """Return 0..1 gust strength at (x, z) — max over all active regions."""
        strongest = 0.0
        for region, current in self._active_gust_strengths():
            dx = x - region.center_x
            dz = z - region.center_z
            if dx * dx + dz * dz > region.radius * region.radius:
                continue
            if current > strongest:
                strongest = current
        return strongest

    def is_hazardous_at(self, x: float, z: float) -> bool:
        """True if (x, z) lies in a gust region currently past the threshold."""
        return self.gust_strength_at(x, z) >= self.GUST_HAZARD_THRESHOLD

    def active_gust_regions(self) -> list[dict]:
        """Return serializable active gust regions for WebSocket broadcast."""
        out: list[dict] = []
        for region, current in self._active_gust_strengths():
            if current < self.GUST_HAZARD_THRESHOLD:
                continue
            out.append(
                {
                    "x": round(region.center_x, 1),
                    "z": round(region.center_z, 1),
                    "radius": round(region.radius, 1),
                    "strength": round(current, 3),
                }
            )
        return out

    def serialize(self) -> dict:
        """Serialize current weather state for WebSocket broadcast."""
        return {
            "wind_speed": round(self._wind_speed, 3),
            "wind_direction": round(self._wind_direction, 4),
            "gusting": self._gusting,
            "gust_regions": self.active_gust_regions(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _active_gust_strengths(self) -> list[tuple[GustRegion, float]]:
        """Return (region, current_strength) pairs for every region.

        current_strength is in [0, base_strength], oscillating on the
        region's own sine cycle. Callers filter by the hazard threshold.
        """
        out: list[tuple[GustRegion, float]] = []
        for region in self._gust_regions:
            # sine in [-1, 1] → max(0, ...) so regions only gust half the time
            raw = math.sin(self._elapsed * (2.0 * math.pi / region.period) + region.phase)
            current = max(0.0, raw) * region.base_strength
            out.append((region, current))
        return out

    @staticmethod
    def _make_gust_regions(seed: int, terrain_size: int) -> tuple[GustRegion, ...]:
        """Seeded placement of 6 gust regions across the world."""
        rng = random.Random(seed ^ 0xAA55)
        n_regions = 6
        regions: list[GustRegion] = []
        # Regions sized to ~8% of world per axis — visible from orbit camera,
        # large enough to be meaningful avoid targets but not world-spanning.
        radius_min = max(200.0, terrain_size * 0.05)
        radius_max = max(radius_min + 100.0, terrain_size * 0.10)
        margin = radius_max
        for _ in range(n_regions):
            cx = rng.uniform(margin, terrain_size - margin)
            cz = rng.uniform(margin, terrain_size - margin)
            r = rng.uniform(radius_min, radius_max)
            strength = rng.uniform(0.7, 1.0)
            period = rng.uniform(60.0, 150.0)  # 1–2.5 minutes per gust cycle
            phase = rng.uniform(0.0, 2.0 * math.pi)
            regions.append(
                GustRegion(
                    center_x=cx,
                    center_z=cz,
                    radius=r,
                    base_strength=strength,
                    period=period,
                    phase=phase,
                )
            )
        return tuple(regions)

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
