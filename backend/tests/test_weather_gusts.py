"""Tests for WeatherSystem gust regions and priority-market avoidance."""

from __future__ import annotations

from src.simulation.weather import WeatherSystem


def test_gust_regions_are_deterministic_for_seed():
    w1 = WeatherSystem(seed=123, terrain_size=10240)
    w2 = WeatherSystem(seed=123, terrain_size=10240)
    assert w1._gust_regions == w2._gust_regions


def test_different_seeds_give_different_regions():
    w1 = WeatherSystem(seed=1, terrain_size=10240)
    w2 = WeatherSystem(seed=2, terrain_size=10240)
    assert w1._gust_regions != w2._gust_regions


def test_gust_regions_inside_world_bounds():
    ts = 10240
    w = WeatherSystem(seed=42, terrain_size=ts)
    for r in w._gust_regions:
        assert 0 <= r.center_x <= ts
        assert 0 <= r.center_z <= ts
        assert r.radius > 0


def test_hazardous_toggles_across_cycle():
    """Each region has its own cycle, so the hazardous set changes over time."""
    w = WeatherSystem(seed=42, terrain_size=10240)
    r = w._gust_regions[0]

    saw_hazardous = False
    saw_safe = False
    for t_sec in range(0, 300, 5):
        w.update(float(t_sec))
        if w.is_hazardous_at(r.center_x, r.center_z):
            saw_hazardous = True
        else:
            saw_safe = True
        if saw_hazardous and saw_safe:
            break
    assert saw_hazardous, "Region never became hazardous in 300s"
    assert saw_safe, "Region never became safe in 300s"


def test_point_outside_all_regions_is_never_hazardous():
    w = WeatherSystem(seed=42, terrain_size=10240)
    # Pick a point far from all regions
    far_x, far_z = -10_000.0, -10_000.0
    for t_sec in range(0, 300, 10):
        w.update(float(t_sec))
        assert not w.is_hazardous_at(far_x, far_z)


def test_active_gust_regions_serialization_filters_threshold():
    w = WeatherSystem(seed=42, terrain_size=10240)
    # Run long enough that at least one region peaks
    max_active = 0
    for t_sec in range(0, 300, 5):
        w.update(float(t_sec))
        active = w.active_gust_regions()
        max_active = max(max_active, len(active))
        for entry in active:
            assert entry["strength"] >= w.GUST_HAZARD_THRESHOLD
            assert {"x", "z", "radius", "strength"} <= entry.keys()
    assert max_active > 0


def test_serialize_includes_gust_regions():
    w = WeatherSystem(seed=42, terrain_size=10240)
    w.update(0.0)
    data = w.serialize()
    assert "gust_regions" in data
    assert isinstance(data["gust_regions"], list)
