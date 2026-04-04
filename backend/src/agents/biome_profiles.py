"""Biome-aware flight profiles for adaptive drone behavior.

Drones adjust altitude, spacing, speed, and detection sensitivity based on
the biome they're searching. This models real-world SAR constraints:
- Forest: fly low and slow, canopy blocks aerial view
- Urban: moderate altitude, buildings create blind spots
- Open terrain (beach, snow): fly high and fast, wide sweeps
"""

from __future__ import annotations

from dataclasses import dataclass

from src.simulation.types import Biome


@dataclass(frozen=True)
class BiomeFlightProfile:
    """Flight parameters optimized for a specific biome."""

    cruise_altitude: float  # meters above terrain surface
    search_spacing: float  # meters between sweep lines
    max_speed: float  # m/s during search
    detection_base_modifier: float  # base detection range multiplier (before altitude adjust)
    repulsion_range: float  # meters — how far drones push apart
    cluster_on_find: bool  # attract nearby drones when a survivor is found
    description: str  # human-readable explanation


# Default profiles per biome
BIOME_PROFILES: dict[int, BiomeFlightProfile] = {
    Biome.WATER.value: BiomeFlightProfile(
        cruise_altitude=60.0,
        search_spacing=40.0,
        max_speed=15.0,
        detection_base_modifier=0.0,
        repulsion_range=30.0,
        cluster_on_find=False,
        description="Water — skip, no survivors",
    ),
    Biome.BEACH.value: BiomeFlightProfile(
        cruise_altitude=60.0,
        search_spacing=35.0,
        max_speed=15.0,
        detection_base_modifier=1.0,
        repulsion_range=30.0,
        cluster_on_find=False,
        description="Beach — open terrain, wide fast sweeps",
    ),
    Biome.FOREST.value: BiomeFlightProfile(
        cruise_altitude=20.0,
        search_spacing=12.0,
        max_speed=8.0,
        detection_base_modifier=0.35,
        repulsion_range=10.0,
        cluster_on_find=True,
        description="Forest — fly low & slow, canopy blocks view, cluster on finds",
    ),
    Biome.URBAN.value: BiomeFlightProfile(
        cruise_altitude=35.0,
        search_spacing=18.0,
        max_speed=10.0,
        detection_base_modifier=0.55,
        repulsion_range=15.0,
        cluster_on_find=True,
        description="Urban — buildings block line-of-sight, moderate altitude",
    ),
    Biome.MOUNTAIN.value: BiomeFlightProfile(
        cruise_altitude=40.0,
        search_spacing=22.0,
        max_speed=10.0,
        detection_base_modifier=0.65,
        repulsion_range=20.0,
        cluster_on_find=False,
        description="Mountain — rocky terrain, variable elevation",
    ),
    Biome.SNOW.value: BiomeFlightProfile(
        cruise_altitude=60.0,
        search_spacing=35.0,
        max_speed=15.0,
        detection_base_modifier=0.95,
        repulsion_range=30.0,
        cluster_on_find=False,
        description="Snow — high contrast, easy to spot survivors",
    ),
}

DEFAULT_PROFILE = BiomeFlightProfile(
    cruise_altitude=50.0,
    search_spacing=25.0,
    max_speed=12.0,
    detection_base_modifier=0.5,
    repulsion_range=20.0,
    cluster_on_find=False,
    description="Default — balanced parameters",
)


def get_profile(biome_value: int) -> BiomeFlightProfile:
    """Get the flight profile for a biome value."""
    return BIOME_PROFILES.get(biome_value, DEFAULT_PROFILE)


def get_profile_at_position(
    x: float, z: float, biome_map: object, width: int, height: int
) -> BiomeFlightProfile:
    """Get the flight profile for a world position by looking up the biome map."""
    import numpy as np

    bm = biome_map
    if not isinstance(bm, np.ndarray):
        return DEFAULT_PROFILE

    col = int(min(max(x, 0), width - 1))
    row = int(min(max(z, 0), height - 1))
    biome_val = int(bm[row, col])
    return get_profile(biome_val)


def compute_altitude_detection_bonus(actual_altitude: float, optimal_altitude: float) -> float:
    """Compute detection modifier based on how close the drone is to optimal altitude.

    Flying at or below optimal altitude (closer to ground) gives better detection.
    Flying higher than optimal degrades detection.

    Returns a multiplier from 0.5 (way too high) to 1.5 (very low, excellent view).
    """
    if optimal_altitude <= 0:
        return 1.0

    ratio = actual_altitude / optimal_altitude

    if ratio <= 0.5:
        # Very low — excellent detection but narrow coverage
        return 1.5
    elif ratio <= 1.0:
        # At or below optimal — good detection
        return 1.0 + 0.5 * (1.0 - ratio)
    elif ratio <= 2.0:
        # Above optimal — detection degrades
        return 1.0 - 0.5 * (ratio - 1.0)
    else:
        # Way too high — poor detection
        return 0.5
