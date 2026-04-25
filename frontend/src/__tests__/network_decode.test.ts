/**
 * Tests for the pure decoding logic in network/types.ts:
 *   - decodeTerrain: base64 + legacy JSON paths produce identical output
 *   - isScenarioBriefing: discriminates between mission and directive briefings
 *
 * These functions sit on the WebSocket boundary — a regression here breaks
 * the entire frontend, so they get explicit unit coverage.
 */

import { describe, expect, it } from "vitest";

import {
  decodeTerrain,
  isScenarioBriefing,
} from "@/network/types";
import type {
  MissionBriefingDirective,
  MissionBriefingScenario,
  TerrainData,
} from "@/network/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Encode a Uint16Array as base64 (matches what the backend sends). */
function uint16ToB64(arr: Uint16Array): string {
  const bytes = new Uint8Array(arr.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function uint8ToB64(arr: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < arr.length; i++) bin += String.fromCharCode(arr[i]);
  return btoa(bin);
}

// ---------------------------------------------------------------------------
// decodeTerrain — base64 path
// ---------------------------------------------------------------------------

describe("decodeTerrain (base64)", () => {
  it("decodes a 2x2 heightmap with full uint16 range", () => {
    // Pixels: 0, 65535/2, 65535/4, 65535 → maps to 0, max/2, max/4, max
    const hm = new Uint16Array([0, 32767, 16383, 65535]);
    const bm = new Uint8Array([0, 1, 2, 3]);
    const data: TerrainData = {
      width: 2,
      height: 2,
      max_elevation: 100.0,
      encoding: "base64_uint16_uint8",
      heightmap_b64: uint16ToB64(hm),
      biome_map_b64: uint8ToB64(bm),
    };

    const decoded = decodeTerrain(data);
    expect(decoded.width).toBe(2);
    expect(decoded.height).toBe(2);
    expect(decoded.max_elevation).toBe(100.0);
    expect(decoded.heightmap.length).toBe(4);
    expect(decoded.biome_map.length).toBe(4);

    // Floats: tolerate small fp error from /65535 division
    expect(decoded.heightmap[0]).toBeCloseTo(0.0, 4);
    expect(decoded.heightmap[1]).toBeCloseTo(50.0, 1);
    expect(decoded.heightmap[2]).toBeCloseTo(25.0, 1);
    expect(decoded.heightmap[3]).toBeCloseTo(100.0, 4);

    expect(Array.from(decoded.biome_map)).toEqual([0, 1, 2, 3]);
  });

  it("returns Float32Array for heightmap (not regular Array)", () => {
    const hm = new Uint16Array([100]);
    const bm = new Uint8Array([5]);
    const data: TerrainData = {
      width: 1,
      height: 1,
      max_elevation: 1.0,
      encoding: "base64_uint16_uint8",
      heightmap_b64: uint16ToB64(hm),
      biome_map_b64: uint8ToB64(bm),
    };
    const decoded = decodeTerrain(data);
    expect(decoded.heightmap).toBeInstanceOf(Float32Array);
    expect(decoded.biome_map).toBeInstanceOf(Uint8Array);
  });
});

// ---------------------------------------------------------------------------
// decodeTerrain — legacy JSON path
// ---------------------------------------------------------------------------

describe("decodeTerrain (legacy JSON)", () => {
  it("decodes 2D arrays in row-major order", () => {
    const data: TerrainData = {
      width: 3,
      height: 2,
      max_elevation: 50.0,
      heightmap: [
        [10, 20, 30],
        [40, 50, 60],
      ],
      biome_map: [
        [0, 1, 2],
        [3, 4, 5],
      ],
    };
    const decoded = decodeTerrain(data);
    expect(Array.from(decoded.heightmap)).toEqual([10, 20, 30, 40, 50, 60]);
    expect(Array.from(decoded.biome_map)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("returns zeroed arrays when neither encoding nor JSON present", () => {
    const data: TerrainData = {
      width: 4,
      height: 3,
      max_elevation: 100.0,
    };
    const decoded = decodeTerrain(data);
    expect(decoded.heightmap.length).toBe(12);
    expect(decoded.biome_map.length).toBe(12);
    expect(decoded.heightmap.every((v) => v === 0)).toBe(true);
    expect(decoded.biome_map.every((v) => v === 0)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// isScenarioBriefing — discriminator
// ---------------------------------------------------------------------------

describe("isScenarioBriefing", () => {
  it("returns true for a scenario briefing (has mission)", () => {
    const msg: MissionBriefingScenario = {
      type: "mission_briefing",
      mission: {
        name: "lost_hiker",
        title: "Lost Hiker",
        description: "Find the missing person.",
        known_facts: ["Started at trailhead"],
        base_position: [500, 50, 500],
        survival_window_seconds: 14400,
        intel_pins: [],
      },
      available: ["lost_hiker", "aircraft_crash"],
    };
    expect(isScenarioBriefing(msg)).toBe(true);
  });

  it("returns false for a directive briefing (no mission, has briefing text)", () => {
    const msg: MissionBriefingDirective = {
      type: "mission_briefing",
      tick: 1234,
      briefing: "Reassigning drones 1, 4 to zone D",
      zone_priorities: { D: "high", A: "low" },
      reasoning: "Operator emphasised eastern quadrant",
    };
    expect(isScenarioBriefing(msg)).toBe(false);
  });
});
