"""Tests for wind_hazard_fn integration in the coordinator + priority market."""

from __future__ import annotations

from src.agents.coordinator import DroneTask, SwarmCoordinator
from src.simulation.engine import create_world
from src.simulation.types import SimConfig, Vec3  # noqa: F401  (used by new test)

_CFG = SimConfig(
    terrain_size=64,
    terrain_seed=42,
    drone_count=3,
    survivor_count=3,
    sensor_failure_prob=0.0,
    comms_failure_prob=0.0,
)


def test_update_without_wind_hazard_fn_is_no_op():
    """Backward-compat: calling update() without wind_hazard_fn still works."""
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)
    commands = coord.update(world, _CFG)
    # Shouldn't raise, shouldn't skip drones
    assert len(commands) > 0


def test_wind_gust_invalidates_targets_inside_it():
    """A drone with a target inside an active gust has its target scrubbed."""
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)
    # First update assigns zones + targets.
    coord.update(world, _CFG)

    # Find a drone whose target is at a known position, then declare that
    # position hazardous.
    target_drone_id = None
    target_xz: tuple[float, float] | None = None
    for drone_id, agent in coord.agents.items():
        if agent.current_target is not None and agent.task != DroneTask.RETURNING_TO_BASE:
            target_drone_id = drone_id
            t = agent.current_target
            target_xz = (t.x, t.z)
            break

    assert target_drone_id is not None, "Expected at least one drone with a target"
    assert target_xz is not None
    tx, tz = target_xz

    def gust_covers_that_target(x: float, z: float) -> bool:
        return abs(x - tx) < 0.5 and abs(z - tz) < 0.5

    coord.update(world, _CFG, wind_hazard_fn=gust_covers_that_target)

    # That specific drone's target should now be None (invalidated).
    assert coord.agents[target_drone_id].current_target is None or (
        coord.agents[target_drone_id].current_target is not None
        and not gust_covers_that_target(
            coord.agents[target_drone_id].current_target.x,
            coord.agents[target_drone_id].current_target.z,
        )
    ), "Target inside active gust was not invalidated"


def test_market_filter_composes_zones_and_wind():
    """is_in_avoid in the coordinator folds both operator avoid zones AND
    wind hazards. If either matches, the market skips the asset."""
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)

    # Register the gust filter — whole eastern half of the map is gusted.
    def east_half_hazardous(x: float, z: float) -> bool:
        return x > world.terrain.width / 2.0

    # Run several ticks under the hazard.
    for _ in range(10):
        coord.update(world, _CFG, wind_hazard_fn=east_half_hazardous)

    # Any drone that got a market assignment should have its target in the
    # western half (or no target at all).
    for drone_id, agent in coord.agents.items():
        t = agent.current_target
        if t is None:
            continue
        assert t.x <= world.terrain.width / 2.0 + 0.5, (
            f"Drone {drone_id} heading into hazardous east half at x={t.x}"
        )


def test_wind_hazard_fn_is_cleared_between_ticks():
    """Passing None on a later tick drops the hazard — drones can re-enter."""
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)
    coord.update(world, _CFG, wind_hazard_fn=lambda x, z: True)  # all hazardous
    coord.update(world, _CFG, wind_hazard_fn=None)
    assert coord._wind_hazard_fn is None


def test_sweep_waypoints_scrubbed_inside_active_gust():
    """Regression: the waypoint-scrub branch inside
    _invalidate_targets_in_wind_gusts previously referenced an undefined
    `inside_avoid` — this test pins the code path by setting up an agent
    with queued sweep waypoints that straddle a gust boundary.
    """
    world = create_world(_CFG)
    coord = SwarmCoordinator(_CFG)
    coord.update(world, _CFG)  # seeds agents + zones

    # Pick any agent and give it a sweep queue straddling a boundary.
    agent = next(iter(coord.agents.values()))
    agent.local_sweep_waypoints = [
        Vec3(10.0, 10.0, 10.0),  # safe (west)
        Vec3(50.0, 10.0, 10.0),  # inside gust (east of x=40)
        Vec3(20.0, 10.0, 20.0),  # safe
        Vec3(60.0, 10.0, 30.0),  # inside gust
    ]
    agent.current_target = None  # so we exercise the waypoint branch, not target branch

    def east_of_40_is_hazardous(x: float, z: float) -> bool:
        return x > 40.0

    coord.update(world, _CFG, wind_hazard_fn=east_of_40_is_hazardous)

    remaining = agent.local_sweep_waypoints or []
    for wp in remaining:
        assert wp.x <= 40.0, f"Waypoint at x={wp.x} should have been scrubbed"
    # Two of four should survive
    assert len(remaining) == 2
