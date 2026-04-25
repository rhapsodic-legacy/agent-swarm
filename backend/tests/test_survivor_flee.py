"""Tests for mobile-survivor flee behavior."""

from __future__ import annotations

import numpy as np

from src.simulation.survivors import FLEE_RADIUS, update_survivors
from src.simulation.types import Biome, Drone, DroneStatus, Survivor, Terrain, Vec3


def _flat_terrain(size: int = 256) -> Terrain:
    heightmap = np.full((size, size), 10.0, dtype=np.float64)
    biome_map = np.full((size, size), Biome.BEACH.value, dtype=np.int32)
    return Terrain(
        width=size,
        height=size,
        max_elevation=200.0,
        heightmap=heightmap,
        biome_map=biome_map,
        seed=42,
    )


def _mobile_survivor(sid: int, x: float, z: float, speed: float = 0.5) -> Survivor:
    return Survivor(id=sid, position=Vec3(x, 10.0, z), mobile=True, speed=speed)


def _drone(did: int, x: float, z: float, status: DroneStatus = DroneStatus.ACTIVE) -> Drone:
    return Drone(id=did, position=Vec3(x, 60.0, z), status=status)


def test_survivor_flees_from_nearby_active_drone():
    """Survivor moves AWAY from a drone within FLEE_RADIUS."""
    terrain = _flat_terrain()
    # Survivor at (128, 128), drone 30m east of it → survivor should move west.
    surv = _mobile_survivor(0, 128.0, 128.0)
    drone = _drone(0, 158.0, 128.0)
    rng = np.random.default_rng(42)

    updated = update_survivors(
        (surv,),
        terrain,
        dt=0.1,
        rng=rng,
        drones=(drone,),
    )
    assert updated[0].position.x < surv.position.x, (
        f"Survivor should flee west; went from x={surv.position.x} to {updated[0].position.x}"
    )


def test_survivor_ignores_failed_drones():
    """FAILED drones don't trigger flee — no rotor noise."""
    terrain = _flat_terrain()
    surv = _mobile_survivor(0, 128.0, 128.0, speed=0.0)  # zero speed — any motion is from flee
    failed = _drone(0, 130.0, 128.0, status=DroneStatus.FAILED)
    rng = np.random.default_rng(42)

    updated = update_survivors(
        (surv,),
        terrain,
        dt=0.1,
        rng=rng,
        drones=(failed,),
    )
    # Speed=0 and no active drones → position unchanged
    assert abs(updated[0].position.x - surv.position.x) < 1e-9
    assert abs(updated[0].position.z - surv.position.z) < 1e-9


def test_survivor_does_not_flee_from_distant_drone():
    """Drone outside FLEE_RADIUS shouldn't affect movement direction."""
    terrain = _flat_terrain()
    surv = _mobile_survivor(0, 128.0, 128.0, speed=0.0)
    # Drone far away — outside flee radius
    distant = _drone(0, 128.0 + FLEE_RADIUS + 50.0, 128.0)
    rng = np.random.default_rng(42)

    updated = update_survivors(
        (surv,),
        terrain,
        dt=0.1,
        rng=rng,
        drones=(distant,),
    )
    # With speed=0 survivor doesn't move at all regardless (wander × speed = 0)
    assert abs(updated[0].position.x - surv.position.x) < 1e-9


def test_fleeing_is_faster_than_wandering():
    """A fleeing survivor covers more ground per tick than a wanderer."""
    terrain = _flat_terrain()
    surv = _mobile_survivor(0, 128.0, 128.0, speed=0.5)

    # No drones → wander. Rng controls direction but not magnitude.
    rng_a = np.random.default_rng(42)
    wandered = update_survivors((surv,), terrain, dt=1.0, rng=rng_a, drones=())
    wander_dist = (
        (wandered[0].position.x - surv.position.x) ** 2
        + (wandered[0].position.z - surv.position.z) ** 2
    ) ** 0.5

    # Drone nearby → flee.
    drone = _drone(0, 128.0 + 40.0, 128.0)
    rng_b = np.random.default_rng(42)
    fled = update_survivors((surv,), terrain, dt=1.0, rng=rng_b, drones=(drone,))
    flee_dist = (
        (fled[0].position.x - surv.position.x) ** 2 + (fled[0].position.z - surv.position.z) ** 2
    ) ** 0.5

    # Flee distance should be approximately FLEE_SPEED_MULT × wander distance,
    # and definitely strictly greater.
    assert flee_dist > wander_dist
    # Expect roughly speed × FLEE_SPEED_MULT × dt = 0.5 × 2.5 × 1 = 1.25
    assert flee_dist > 1.0


def test_discovered_survivors_do_not_flee():
    """Once found, survivors stop moving entirely."""
    terrain = _flat_terrain()
    surv = Survivor(
        id=0,
        position=Vec3(128.0, 10.0, 128.0),
        mobile=True,
        speed=0.5,
        discovered=True,
    )
    drone = _drone(0, 138.0, 128.0)
    rng = np.random.default_rng(42)

    updated = update_survivors((surv,), terrain, dt=1.0, rng=rng, drones=(drone,))
    # Frozen — exact same object is returned
    assert updated[0] is surv


def test_flee_at_radius_boundary_triggers():
    """Right at FLEE_RADIUS - epsilon the survivor should flee."""
    terrain = _flat_terrain()
    surv = _mobile_survivor(0, 128.0, 128.0, speed=0.5)
    drone = _drone(0, 128.0 + FLEE_RADIUS - 1.0, 128.0)
    rng = np.random.default_rng(42)

    updated = update_survivors((surv,), terrain, dt=0.1, rng=rng, drones=(drone,))
    # Should move west (away from drone at +x)
    assert updated[0].position.x < surv.position.x


def test_update_survivors_without_drones_kwarg_works():
    """Backwards-compat: existing callers that don't pass drones still work."""
    terrain = _flat_terrain()
    surv = _mobile_survivor(0, 128.0, 128.0)
    rng = np.random.default_rng(42)

    updated = update_survivors((surv,), terrain, dt=0.1, rng=rng)
    assert len(updated) == 1
