"""Tests for the Bayesian search map (PoC grid + updates).

These test the math directly — no sim loop, no WebSocket. If the math is
wrong here, everything downstream is wrong.
"""

from __future__ import annotations

import numpy as np

from src.simulation.search_map import SearchMap

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_has_zero_mass(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        assert sm.total_mass() == 0.0
        assert sm.poc.shape == (256, 256)

    def test_uniform_has_unit_mass(self):
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        assert abs(sm.total_mass() - 1.0) < 1e-5

    def test_uniform_cells_are_equal(self):
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        cells = sm.poc.ravel()
        assert cells.min() == cells.max()

    def test_cell_size_determines_grid(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=20.0)
        assert sm.poc.shape == (512, 512)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


class TestCoordinates:
    def test_world_to_cell_roundtrip(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        col, row = sm.world_to_cell(5120.0, 5120.0)
        # 5120m / 40m = 128 cell index (center of 256-wide grid)
        assert col == 128
        assert row == 128

    def test_cell_to_world_center(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        x, z = sm.cell_to_world(0, 0)
        # Cell 0 center is at (20, 20) — half a cell in
        assert x == 20.0
        assert z == 20.0

    def test_coords_clamped_to_grid(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        col, row = sm.world_to_cell(-100.0, 99999.0)
        assert col == 0
        assert row == 255  # clamped to grid-1


# ---------------------------------------------------------------------------
# Prior seeding
# ---------------------------------------------------------------------------


class TestPriorSeeding:
    def test_gaussian_peaks_at_center(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_gaussian(center_world=(5000, 5000), radius_meters=400, weight=1.0)
        col, row = sm.world_to_cell(5000, 5000)
        peak = float(sm.poc[row, col])
        # Peak should be close to 1.0 (the weight)
        assert peak > 0.95

    def test_gaussian_falls_off_with_distance(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_gaussian(center_world=(5000, 5000), radius_meters=400, weight=1.0)
        col, row = sm.world_to_cell(5000, 5000)
        peak = float(sm.poc[row, col])
        # 10 cells away (400m) is 1σ — should be ~0.6 of peak
        edge = float(sm.poc[row, col + 10])
        assert 0.5 * peak < edge < 0.75 * peak

    def test_gaussian_is_additive(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_gaussian(center_world=(5000, 5000), radius_meters=400, weight=1.0)
        mass_after_first = sm.total_mass()
        sm.add_gaussian(center_world=(5000, 5000), radius_meters=400, weight=1.0)
        mass_after_second = sm.total_mass()
        # Second add on the same spot should roughly double mass
        assert 1.8 * mass_after_first < mass_after_second < 2.2 * mass_after_first

    def test_uniform_rect_has_uniform_values(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_uniform_rect(0, 0, 2000, 2000, weight=1.0)
        # All cells in the rect should have value 1.0
        assert sm.poc[0, 0] == 1.0
        # Cells outside should be zero
        assert sm.poc[100, 100] == 0.0

    def test_normalize_sets_total_mass(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_gaussian(center_world=(5000, 5000), radius_meters=400, weight=5.0)
        sm.normalize(target_mass=1.0)
        assert abs(sm.total_mass() - 1.0) < 1e-4

    def test_normalize_handles_empty_gracefully(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.normalize()  # should not crash
        assert sm.total_mass() == 0.0


# ---------------------------------------------------------------------------
# Bayesian update math — the critical part
# ---------------------------------------------------------------------------


class TestBayesianUpdate:
    def test_failed_scan_decreases_scanned_cells(self):
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        col, row = sm.world_to_cell(5000, 5000)
        before = float(sm.poc[row, col])
        sm.update_after_failed_scan(
            center_world=(5000, 5000),
            radius_meters=100,
            pod=0.5,
        )
        after = float(sm.poc[row, col])
        assert after < before

    def test_failed_scan_leaves_unscanned_cells_untouched(self):
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        # A cell far from the scan
        far_col, far_row = sm.world_to_cell(1000, 1000)
        before = float(sm.poc[far_row, far_col])
        sm.update_after_failed_scan(
            center_world=(5000, 5000),
            radius_meters=100,
            pod=0.5,
        )
        after = float(sm.poc[far_row, far_col])
        assert after == before

    def test_failed_scan_with_pod_zero_does_nothing(self):
        sm = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        before = sm.poc.copy()
        sm.update_after_failed_scan(
            center_world=(5000, 5000),
            radius_meters=100,
            pod=0.0,
        )
        np.testing.assert_array_equal(sm.poc, before)

    def test_failed_scan_formula_matches_koopman(self):
        """P' = P * (1 - d) / (1 - P * d) — the standard search-theory formula."""
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        # Set a single cell to known value
        col, row = sm.world_to_cell(5000, 5000)
        sm.poc[row, col] = 0.5  # 50% prior the target is here
        pod = 0.8
        sm.update_after_failed_scan(
            center_world=(5000, 5000),
            radius_meters=20,
            pod=pod,
        )
        expected = 0.5 * (1 - pod) / (1 - 0.5 * pod)  # = 0.1 / 0.6 = 0.1667
        actual = float(sm.poc[row, col])
        assert abs(actual - expected) < 1e-4

    def test_repeated_scans_drive_poc_toward_zero(self):
        """If we keep scanning and failing, the cell's PoC should approach 0."""
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        col, row = sm.world_to_cell(5000, 5000)
        sm.poc[row, col] = 0.9
        for _ in range(20):
            sm.update_after_failed_scan(
                center_world=(5000, 5000),
                radius_meters=20,
                pod=0.5,
            )
        assert float(sm.poc[row, col]) < 0.01

    def test_high_pod_decreases_more_than_low_pod(self):
        sm_lo = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        sm_hi = SearchMap.uniform(world_size=10240.0, cell_size=40.0)
        col, row = sm_lo.world_to_cell(5000, 5000)
        sm_lo.update_after_failed_scan((5000, 5000), 100, pod=0.2)
        sm_hi.update_after_failed_scan((5000, 5000), 100, pod=0.9)
        assert sm_hi.poc[row, col] < sm_lo.poc[row, col]


# ---------------------------------------------------------------------------
# Hottest-cells query (used by coordinator for drone targeting)
# ---------------------------------------------------------------------------


class TestHottestCells:
    def test_returns_highest_values(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.poc[10, 20] = 0.5
        sm.poc[50, 60] = 0.9
        sm.poc[100, 100] = 0.3
        hot = sm.hottest_cells(2)
        assert len(hot) == 2
        # First should be the highest
        col0, row0, val0 = hot[0]
        assert (col0, row0) == (60, 50)
        assert abs(val0 - 0.9) < 1e-6

    def test_exclude_set_is_respected(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.poc[10, 20] = 0.5
        sm.poc[50, 60] = 0.9
        hot = sm.hottest_cells(1, exclude_cells={(60, 50)})
        assert len(hot) == 1
        col, row, val = hot[0]
        assert (col, row) == (20, 10)

    def test_empty_map_returns_empty(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        # All zeros — still returns the top-n but all zero-valued
        hot = sm.hottest_cells(5)
        assert len(hot) == 5
        assert all(val == 0.0 for _, _, val in hot)


# ---------------------------------------------------------------------------
# Downsample for serialization
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_downsample_preserves_hotspot(self):
        """Hotspots must survive downsampling — averaging would hide them."""
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        # Single very hot cell
        sm.poc[100, 100] = 1.0
        down = sm.downsample(target_size=64)
        # The hotspot should still be visible after downsampling
        assert down.max() == 1.0

    def test_downsample_shape(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        down = sm.downsample(target_size=64)
        assert down.shape == (64, 64)

    def test_downsample_to_larger_returns_full(self):
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        down = sm.downsample(target_size=999)
        assert down.shape == sm.poc.shape


# ---------------------------------------------------------------------------
# Integration: realistic SAR scenario
# ---------------------------------------------------------------------------


class TestRealisticScenario:
    def test_drone_clearing_area_reduces_poc_in_path(self):
        """Simulate a drone searching a known-hot area — PoC should drop
        at the scanned cells even though the global-max moves to the
        unscanned Gaussian tail."""
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_gaussian((5000, 5000), radius_meters=400, weight=1.0)
        sm.normalize()

        initial_mass = sm.total_mass()
        col, row = sm.world_to_cell(5000, 5000)
        initial_center = float(sm.poc[row, col])

        # Drone flies through and scans the hot region multiple times.
        # Scan radius is 100m (~2.5 cells) — small compared to the
        # Gaussian's 400m sigma, so only the central cells are updated.
        for _ in range(10):
            sm.update_after_failed_scan((5000, 5000), radius_meters=100, pod=0.3)

        final_mass = sm.total_mass()
        final_center = float(sm.poc[row, col])

        # Mass should decrease (we're "consuming" probability by not finding)
        assert final_mass < initial_mass
        # The CENTER cell (scanned every iteration) must drop substantially.
        # After 10 scans with PoD=0.3, expected ratio ≈ 0.7^10 ≈ 0.028
        assert final_center < initial_center * 0.1

    def test_hottest_cell_matches_gaussian_peak(self):
        """After seeding a Gaussian, the hottest cell should be its center."""
        sm = SearchMap.empty(world_size=10240.0, cell_size=40.0)
        sm.add_gaussian(center_world=(3000, 7000), radius_meters=300, weight=1.0)
        hot = sm.hottest_cells(1)
        col, row, _ = hot[0]
        x, z = sm.cell_to_world(col, row)
        # Should match the gaussian center within 1 cell
        assert abs(x - 3000) < 80
        assert abs(z - 7000) < 80
