"""Scenario presets for quick configuration."""

from __future__ import annotations

from src.simulation.types import SimConfig

PRESETS: dict[str, SimConfig] = {
    "small": SimConfig(
        terrain_size=64,
        drone_count=8,
        survivor_count=6,
        drone_battery_drain_rate=0.08,
        sensor_failure_prob=0.0001,
    ),
    "medium": SimConfig(
        terrain_size=128,
        drone_count=20,
        survivor_count=15,
    ),
    "large": SimConfig(
        terrain_size=256,
        drone_count=40,
        survivor_count=30,
        drone_battery_drain_rate=0.03,
    ),
    "stress": SimConfig(
        terrain_size=256,
        drone_count=80,
        survivor_count=50,
        drone_battery_drain_rate=0.06,
        sensor_failure_prob=0.0002,
        comms_failure_prob=0.0001,
    ),
    "huge": SimConfig(
        terrain_size=1024,
        drone_count=80,
        survivor_count=100,
        drone_battery_drain_rate=0.02,
        sensor_failure_prob=0.0001,
    ),
}


def get_preset(name: str) -> SimConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        msg = f"Unknown preset '{name}'. Available: {available}"
        raise ValueError(msg)
    return PRESETS[name]
