"""Test non-maximum suppression for hotspot diversity."""

from __future__ import annotations

import math

from src.simulation.search_map import SearchMap


def test_diverse_hotspots_returns_separated_cells():
    sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
    # Place three hot clusters, one dominant
    sm.add_gaussian((2000, 2000), radius_meters=200, weight=1.0)  # dominant
    sm.add_gaussian((5000, 5000), radius_meters=200, weight=0.5)  # medium
    sm.add_gaussian((8000, 8000), radius_meters=200, weight=0.3)  # weak

    # Ask for 3 hotspots with 1000m separation
    hotspots = sm.diverse_hotspots(n=3, min_separation_meters=1000)
    assert len(hotspots) == 3

    # Each should be at a different cluster
    positions = [sm.cell_to_world(c, r) for c, r, _ in hotspots]
    # All pairwise distances should exceed 1000m
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dz = positions[i][1] - positions[j][1]
            dist = math.sqrt(dx * dx + dz * dz)
            assert dist >= 1000.0, f"Hotspots {i} and {j} are only {dist:.0f}m apart (need ≥1000m)"


def test_diverse_hotspots_returns_in_priority_order():
    sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
    sm.add_gaussian((5000, 5000), radius_meters=200, weight=0.3)
    sm.add_gaussian((2000, 2000), radius_meters=200, weight=1.0)  # dominant
    sm.add_gaussian((8000, 8000), radius_meters=200, weight=0.5)

    hotspots = sm.diverse_hotspots(n=3, min_separation_meters=1000)
    # First should be the dominant one (near 2000, 2000)
    col0, row0, _ = hotspots[0]
    x0, z0 = sm.cell_to_world(col0, row0)
    assert abs(x0 - 2000) < 100
    assert abs(z0 - 2000) < 100


def test_diverse_hotspots_fewer_than_requested():
    """With only two clusters, asking for 5 hotspots should return 2-ish."""
    sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
    sm.add_gaussian((2000, 2000), radius_meters=100, weight=1.0)
    sm.add_gaussian((8000, 8000), radius_meters=100, weight=1.0)

    hotspots = sm.diverse_hotspots(n=5, min_separation_meters=2000)
    # Should not return more than a handful even though we asked for 5
    assert 2 <= len(hotspots) <= 5
    # All should be separated by at least 2000m
    positions = [sm.cell_to_world(c, r) for c, r, _ in hotspots]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dz = positions[i][1] - positions[j][1]
            assert math.sqrt(dx * dx + dz * dz) >= 2000.0


def test_diverse_hotspots_empty_map():
    sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
    hotspots = sm.diverse_hotspots(n=3, min_separation_meters=1000)
    assert hotspots == []


def test_diverse_hotspots_respects_n():
    sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
    sm.add_gaussian((2000, 2000), radius_meters=200, weight=1.0)
    sm.add_gaussian((5000, 5000), radius_meters=200, weight=0.5)
    sm.add_gaussian((8000, 8000), radius_meters=200, weight=0.3)

    hotspots = sm.diverse_hotspots(n=2, min_separation_meters=1000)
    assert len(hotspots) == 2
