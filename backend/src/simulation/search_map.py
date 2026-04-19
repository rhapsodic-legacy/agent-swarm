"""Bayesian search map — Probability of Containment (PoC) grid.

This is the mathematical heart of real search-and-rescue planning, based on
Koopman / Stone search theory and the US Coast Guard's CASP methodology.

Three grids sit on top of the world:
    PoC (Probability of Containment): likelihood the target is in each cell
    PoD (Probability of Detection):   sensor effectiveness per cell, per scan
    PoS (Probability of Success):     PoC × PoD — what a scan "buys us"

The coordinator's job is to route drones to maximize integrated PoS over the
mission's remaining resources (time, battery, drone-count).

Classic failure-to-detect Bayesian update (when a cell is searched and nothing
is found):
    P'(cell i scanned)     = P(i) * (1 - d(i)) / N
    P'(cell j not scanned) = P(j) / N
    N = 1 - Σ P(i) * d(i)    over all scanned cells

Evidence updates (Phase 3) will be implemented separately as multiplicative
weight maps applied to the prior + posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchMap:
    """Probability of Containment grid for a search mission.

    The grid is stored as float32 — at 256x256 resolution (40m/cell for a 10km
    world) that's 256KB, cheap to copy and serialize. At 512x512 (20m/cell)
    it's 1MB, still manageable.

    Attributes:
        poc: 2D array of float32, shape (rows, cols). Values in [0, 1].
             Sum of the grid is the total probability mass ("the target is
             somewhere in the grid"). Typically starts at 1.0 if the target
             is definitely within the search area.
        cell_size: Meters per cell. World is `cell_size * rows` meters tall,
                   `cell_size * cols` meters wide.
        world_size: Total world size in meters (square world assumed).
    """

    poc: np.ndarray
    cell_size: float
    world_size: float

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, world_size: float, cell_size: float = 40.0) -> SearchMap:
        """Create an empty (zero) map. Caller must seed a prior."""
        n = int(round(world_size / cell_size))
        return cls(
            poc=np.zeros((n, n), dtype=np.float32),
            cell_size=cell_size,
            world_size=world_size,
        )

    @classmethod
    def uniform(cls, world_size: float, cell_size: float = 40.0) -> SearchMap:
        """Create a uniform prior — every cell equally likely. Total mass = 1."""
        sm = cls.empty(world_size, cell_size)
        sm.poc[:, :] = 1.0 / sm.poc.size
        return sm

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        """Convert world (x, z) meters → (col, row). Clamps to grid bounds."""
        n = self.poc.shape[0]
        col = max(0, min(n - 1, int(x / self.cell_size)))
        row = max(0, min(n - 1, int(z / self.cell_size)))
        return col, row

    def cell_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Convert (col, row) → world (x, z) meters at cell center."""
        return (col + 0.5) * self.cell_size, (row + 0.5) * self.cell_size

    # ------------------------------------------------------------------
    # Prior seeding
    # ------------------------------------------------------------------

    def add_gaussian(
        self,
        center_world: tuple[float, float],
        radius_meters: float,
        weight: float,
    ) -> None:
        """Add a Gaussian blob of probability at the given world location.

        Multiple calls stack — call this once per cluster/hotspot, then
        normalize().

        Args:
            center_world: (x, z) in meters
            radius_meters: 1-sigma radius of the Gaussian
            weight: peak additive probability mass at the center
        """
        n = self.poc.shape[0]
        cx_cell, cz_cell = self.world_to_cell(*center_world)
        sigma = max(radius_meters / self.cell_size, 0.5)

        # Only compute over a patch within 3σ of the center
        patch_r = int(math.ceil(sigma * 3.5))
        r_min = max(0, cz_cell - patch_r)
        r_max = min(n, cz_cell + patch_r + 1)
        c_min = max(0, cx_cell - patch_r)
        c_max = min(n, cx_cell + patch_r + 1)

        if r_min >= r_max or c_min >= c_max:
            return

        rr = np.arange(r_min, r_max, dtype=np.float32)[:, None]
        cc = np.arange(c_min, c_max, dtype=np.float32)[None, :]
        dist_sq = (rr - cz_cell) ** 2 + (cc - cx_cell) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)

        self.poc[r_min:r_max, c_min:c_max] += gaussian * np.float32(weight)

    def add_uniform_rect(
        self,
        x_min: float,
        z_min: float,
        x_max: float,
        z_max: float,
        weight: float,
    ) -> None:
        """Add a uniform rectangle of probability (for known search-area bounds)."""
        col_min, row_min = self.world_to_cell(x_min, z_min)
        col_max, row_max = self.world_to_cell(x_max, z_max)
        col_max = max(col_max, col_min + 1)
        row_max = max(row_max, row_min + 1)
        self.poc[row_min:row_max, col_min:col_max] += np.float32(weight)

    def normalize(self, target_mass: float = 1.0) -> None:
        """Rescale so total probability mass equals target_mass.

        Call once after seeding all priors. If target_mass < 1.0, it means
        there's some probability the target is outside the search area.
        """
        total = float(self.poc.sum())
        if total > 1e-9:
            self.poc *= np.float32(target_mass / total)

    # ------------------------------------------------------------------
    # Bayesian update after scanning
    # ------------------------------------------------------------------

    def update_after_failed_scan(
        self,
        center_world: tuple[float, float],
        radius_meters: float,
        pod: float,
    ) -> float:
        """Bayesian update when drone scanned an area and found nothing.

        For each cell within `radius_meters` of center_world:
            PoC_new = PoC_old * (1 - PoD) / (1 - PoC_old * PoD)

        This is the "failure-to-detect" update — cells that were searched
        without success have their PoC reduced. Cells outside the scan are
        unchanged.

        Args:
            center_world: Scan center (x, z) in meters
            radius_meters: Scan radius
            pod: Probability of detection in [0, 1] — sensor effectiveness
                 given the target is in a scanned cell

        Returns:
            The amount of probability mass "consumed" (useful for tracking
            search effectiveness over time).
        """
        if pod <= 0.0:
            return 0.0
        pod = min(pod, 0.999)  # avoid division by zero when pod = 1.0

        n = self.poc.shape[0]
        cx_cell, cz_cell = self.world_to_cell(*center_world)
        radius_cells = max(1, int(math.ceil(radius_meters / self.cell_size)))

        r_min = max(0, cz_cell - radius_cells)
        r_max = min(n, cz_cell + radius_cells + 1)
        c_min = max(0, cx_cell - radius_cells)
        c_max = min(n, cx_cell + radius_cells + 1)

        if r_min >= r_max or c_min >= c_max:
            return 0.0

        # Build circular mask within the bounding box
        rr = np.arange(r_min, r_max, dtype=np.float32)[:, None]
        cc = np.arange(c_min, c_max, dtype=np.float32)[None, :]
        dist_sq = (rr - cz_cell) ** 2 + (cc - cx_cell) ** 2
        mask = dist_sq <= radius_cells * radius_cells

        patch = self.poc[r_min:r_max, c_min:c_max]
        mass_before = float(patch[mask].sum())

        # Bayesian failure-to-detect: PoC' = PoC * (1 - PoD) / (1 - PoC * PoD)
        inv_pod = np.float32(1.0 - pod)
        denom = 1.0 - patch * np.float32(pod)
        updated = patch * inv_pod / np.maximum(denom, np.float32(1e-9))

        # Apply only to masked (in-circle) cells
        patch_new = np.where(mask, updated, patch)
        self.poc[r_min:r_max, c_min:c_max] = patch_new

        mass_after = float(patch_new[mask].sum())
        return mass_before - mass_after

    # ------------------------------------------------------------------
    # Posterior update from evidence
    # ------------------------------------------------------------------

    def update_on_evidence(
        self,
        position: tuple[float, float],
        kind: str,
        *,
        confidence: float = 0.7,
        heading: float | None = None,
    ) -> float:
        """Re-shape the posterior after a drone discovers evidence.

        Evidence is a multiplicative weight map overlaid on the current
        PoC, then renormalized. Cells the evidence *supports* get boosted
        (multiplier > 1); other cells get relatively deprecated.

        The weight map geometry depends on the evidence kind:

          * footprint: a forward cone along `heading`. Cells inside the
            cone and within reach-distance get boosted; cells behind the
            footprint (opposite heading) get mildly suppressed.
          * debris: an isotropic ring at the expected drift distance.
            Strongest boost at ring radius, falling off inside and outside.
          * signal_fire: a tight high-confidence circle (active survivor
            signaling — we're very close).

        Returns the multiplicative strength actually applied (peak weight,
        useful for logging/testing).

        Args:
            position: (x, z) world-space location of the evidence.
            kind: EvidenceKind value ("footprint" / "debris" / "signal_fire").
            confidence: 0-1 scale factor. 1.0 = strongest update; 0.0 = no-op.
            heading: radians, 0 = +Z/north, clockwise. Required for
                directional kinds (footprint); ignored otherwise.

        Returns:
            Peak weight applied (≥1.0 on updates, 0.0 on no-op).
        """
        if confidence <= 0.0:
            return 0.0

        weight = self._evidence_weight_map(position, kind, confidence, heading)
        if weight is None:
            return 0.0

        # Multiplicative update + renormalize. Unlike failure-to-detect,
        # evidence *redistributes* belief — total mass stays at 1.0, but
        # the shape of the posterior changes.
        before_mass = float(self.poc.sum())
        self.poc = self.poc * weight
        after_mass = float(self.poc.sum())
        if after_mass > 1e-9 and before_mass > 1e-9:
            self.poc *= np.float32(before_mass / after_mass)
        return float(weight.max())

    def _evidence_weight_map(
        self,
        position: tuple[float, float],
        kind: str,
        confidence: float,
        heading: float | None,
    ) -> np.ndarray | None:
        """Build the multiplicative weight grid for an evidence update.

        The grid has shape self.poc.shape and dtype float32. Values at 1.0
        mean "no change at this cell"; > 1.0 means boost; < 1.0 means
        suppress. The caller multiplies PoC by this and renormalizes.

        Returns None if the evidence falls outside the grid entirely.
        """
        weight = np.ones_like(self.poc, dtype=np.float32)

        px, pz = position
        if not (0.0 <= px < self.world_size and 0.0 <= pz < self.world_size):
            return None

        if kind == "footprint":
            # Directional cone: 800m radius, ±35° half-angle, boosted in
            # the heading direction.
            if heading is None:
                heading = 0.0
            self._apply_cone(
                weight, px, pz, heading,
                radius_m=800.0, half_angle=0.61,
                peak=1.0 + 3.0 * confidence,
            )
        elif kind == "debris":
            # Ring at ~400m, peak boost 2.5× at the ring, falling off.
            self._apply_ring(
                weight, px, pz,
                ring_radius_m=400.0, sigma_m=200.0,
                peak=1.0 + 1.5 * confidence,
            )
        elif kind == "signal_fire":
            # Tight circle: 500m radius, very high boost at center
            # (active survivor signaling → they're right there).
            self._apply_gaussian_boost(
                weight, px, pz, radius_m=500.0, peak=1.0 + 6.0 * confidence,
            )
        else:
            # Unknown kind — isotropic moderate boost.
            self._apply_gaussian_boost(
                weight, px, pz, radius_m=400.0, peak=1.0 + 1.0 * confidence,
            )

        return weight

    def _apply_gaussian_boost(
        self,
        weight: np.ndarray,
        x: float,
        z: float,
        radius_m: float,
        peak: float,
    ) -> None:
        """Add a Gaussian bump (peak at center, 1.0 at infinity) to `weight`."""
        n = weight.shape[0]
        cx, cz = self.world_to_cell(x, z)
        sigma = max(radius_m / self.cell_size, 1.0)
        patch_r = int(math.ceil(sigma * 3.0))
        r_min = max(0, cz - patch_r)
        r_max = min(n, cz + patch_r + 1)
        c_min = max(0, cx - patch_r)
        c_max = min(n, cx + patch_r + 1)
        if r_min >= r_max or c_min >= c_max:
            return

        rr = np.arange(r_min, r_max, dtype=np.float32)[:, None]
        cc = np.arange(c_min, c_max, dtype=np.float32)[None, :]
        dist_sq = (rr - cz) ** 2 + (cc - cx) ** 2
        bump = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32) * np.float32(peak - 1.0)
        weight[r_min:r_max, c_min:c_max] += bump

    def _apply_cone(
        self,
        weight: np.ndarray,
        x: float,
        z: float,
        heading: float,
        radius_m: float,
        half_angle: float,
        peak: float,
    ) -> None:
        """Boost cells inside a forward cone; mildly suppress cells behind.

        Heading uses the project convention: 0 = +Z/north, clockwise.
        Vector components: (sin(h), cos(h)) in (x, z).
        """
        n = weight.shape[0]
        cx, cz = self.world_to_cell(x, z)
        radius_cells = int(math.ceil(radius_m / self.cell_size))
        r_min = max(0, cz - radius_cells)
        r_max = min(n, cz + radius_cells + 1)
        c_min = max(0, cx - radius_cells)
        c_max = min(n, cx + radius_cells + 1)
        if r_min >= r_max or c_min >= c_max:
            return

        rr = np.arange(r_min, r_max, dtype=np.float32)[:, None]
        cc = np.arange(c_min, c_max, dtype=np.float32)[None, :]

        # Vectors from (cx, cz) to each cell, in (x, z) = (col, row).
        dx = cc - cx
        dz = rr - cz
        dist = np.sqrt(dx * dx + dz * dz)
        # Avoid div-by-zero at the origin cell
        dist_safe = np.maximum(dist, 1e-6)

        # Heading direction vector (x, z) = (sin(h), cos(h))
        hx = math.sin(heading)
        hz = math.cos(heading)
        # dot(normalized_cell_vec, heading_vec)
        cos_angle = (dx * hx + dz * hz) / dist_safe

        # Forward cone: inside the half-angle AND within radius.
        half_cos = math.cos(half_angle)
        in_cone = (cos_angle >= half_cos) & (dist <= radius_cells)
        # Falloff with distance (linear from peak at origin to 1.0 at radius).
        falloff = np.clip(1.0 - dist / max(radius_cells, 1.0), 0.0, 1.0)
        cone_boost = in_cone.astype(np.float32) * falloff * np.float32(peak - 1.0)

        # Behind: mild suppression (multiplier < 1). Cells with cos_angle
        # below -half_cos are "behind" the footprint.
        behind = (cos_angle <= -half_cos) & (dist <= radius_cells)
        behind_suppress = behind.astype(np.float32) * falloff * np.float32(-0.4)

        patch = weight[r_min:r_max, c_min:c_max]
        # Multiplicative on the boosted side, additive suppression on the
        # behind side. We add to patch (starts at 1.0):
        #   boost > 0 → pushes multiplier above 1
        #   suppress < 0 → pushes multiplier toward ~0.6
        patch += cone_boost
        patch += behind_suppress
        # Floor so we never zero out a cell entirely — prior belief should
        # never be annihilated by a single piece of evidence.
        np.maximum(patch, np.float32(0.2), out=patch)
        weight[r_min:r_max, c_min:c_max] = patch

    def _apply_ring(
        self,
        weight: np.ndarray,
        x: float,
        z: float,
        ring_radius_m: float,
        sigma_m: float,
        peak: float,
    ) -> None:
        """Apply a Gaussian ring (peak at ring_radius, falls off either side)."""
        n = weight.shape[0]
        cx, cz = self.world_to_cell(x, z)
        outer_r_cells = int(math.ceil((ring_radius_m + 3.0 * sigma_m) / self.cell_size))
        r_min = max(0, cz - outer_r_cells)
        r_max = min(n, cz + outer_r_cells + 1)
        c_min = max(0, cx - outer_r_cells)
        c_max = min(n, cx + outer_r_cells + 1)
        if r_min >= r_max or c_min >= c_max:
            return

        ring_cells = ring_radius_m / self.cell_size
        sigma_cells = max(sigma_m / self.cell_size, 1.0)

        rr = np.arange(r_min, r_max, dtype=np.float32)[:, None]
        cc = np.arange(c_min, c_max, dtype=np.float32)[None, :]
        dist = np.sqrt((rr - cz) ** 2 + (cc - cx) ** 2)
        dist_from_ring = dist - np.float32(ring_cells)
        ring_profile = np.exp(
            -(dist_from_ring * dist_from_ring) / (2.0 * sigma_cells * sigma_cells)
        )
        boost = ring_profile.astype(np.float32) * np.float32(peak - 1.0)
        weight[r_min:r_max, c_min:c_max] += boost

    # ------------------------------------------------------------------
    # Posterior inspection
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        """Sum of all PoC values — remaining 'probability the target exists'."""
        return float(self.poc.sum())

    def hottest_cells(
        self,
        n: int,
        exclude_cells: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Return the top-n cells by PoC value, optionally excluding some.

        Returns list of (col, row, poc_value). When we need several drones
        to target different hot spots, the coordinator calls this and
        excludes already-assigned cells on subsequent calls.
        """
        flat = self.poc.ravel()
        cols = self.poc.shape[1]

        if exclude_cells:
            # Mask out excluded cells by setting them to -inf
            flat = flat.copy()
            for (ex_col, ex_row) in exclude_cells:
                if 0 <= ex_row < self.poc.shape[0] and 0 <= ex_col < cols:
                    flat[ex_row * cols + ex_col] = -np.inf

        n = min(n, flat.size)
        if n <= 0:
            return []

        # argpartition + argsort for top-n
        top_indices = np.argpartition(flat, -n)[-n:]
        top_indices = top_indices[np.argsort(-flat[top_indices])]

        results: list[tuple[int, int, float]] = []
        for idx in top_indices:
            val = float(flat[idx])
            if not math.isfinite(val):
                continue
            row = int(idx) // cols
            col = int(idx) % cols
            results.append((col, row, val))
        return results

    def diverse_hotspots(
        self,
        n: int,
        min_separation_meters: float,
    ) -> list[tuple[int, int, float]]:
        """Return up to n hotspots with minimum geographic separation.

        Uses non-maximum suppression: pick the hottest cell, blacklist all
        cells within min_separation, pick next-hottest remaining, repeat.
        This gives drones targets from DIFFERENT clusters instead of N targets
        all from the same hotspot.

        Returns list of (col, row, poc_value), sorted by value descending.
        """
        if n <= 0 or self.poc.size == 0:
            return []

        sep_cells = max(1, int(min_separation_meters / self.cell_size))
        sep_sq = sep_cells * sep_cells

        # Copy poc so we can zero-out regions during NMS
        working = self.poc.copy()
        rows, cols = working.shape
        result: list[tuple[int, int, float]] = []

        for _ in range(n):
            if working.size == 0:
                break
            flat_idx = int(working.argmax())
            val = float(working.flat[flat_idx])
            if val <= 0.0:
                break
            row = flat_idx // cols
            col = flat_idx % cols
            result.append((col, row, val))

            # Zero out a box around this hotspot (cheaper than circle, close enough)
            r_min = max(0, row - sep_cells)
            r_max = min(rows, row + sep_cells + 1)
            c_min = max(0, col - sep_cells)
            c_max = min(cols, col + sep_cells + 1)
            rr = np.arange(r_min, r_max)[:, None]
            cc = np.arange(c_min, c_max)[None, :]
            mask = (rr - row) ** 2 + (cc - col) ** 2 <= sep_sq
            working[r_min:r_max, c_min:c_max] = np.where(
                mask, np.float32(0.0), working[r_min:r_max, c_min:c_max]
            )

        return result

    def downsample(self, target_size: int) -> np.ndarray:
        """Return a downsampled view for serialization to frontend.

        Uses block-max to preserve hotspots (averaging would hide peaks).
        """
        src = self.poc.shape[0]
        if target_size >= src:
            return self.poc.copy()

        block = src // target_size
        if block < 1:
            return self.poc.copy()
        trimmed = self.poc[: block * target_size, : block * target_size]
        return trimmed.reshape(target_size, block, target_size, block).max(axis=(1, 3))
