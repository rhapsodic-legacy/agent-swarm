/**
 * Unit tests for chunked terrain rendering and fog-of-war.
 *
 * These tests validate logic that caused visual bugs in production:
 *   - Chunk data with `resolution` field must use resolution (not size) for grid dimensions
 *   - FogOfWarRenderer must track initialization state correctly
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ChunkTerrainData } from "@/network/types";

// ---------------------------------------------------------------------------
// 1. Chunk resolution field is used for grid dimensions
// ---------------------------------------------------------------------------

describe("chunk terrain resolution handling", () => {
  it("uses resolution (not size) for data array indexing", () => {
    // Mock a chunk as the backend sends it: size=1024, resolution=256 (downsampled)
    const chunk: ChunkTerrainData = {
      type: "chunk_terrain",
      cx: 0,
      cz: 0,
      origin_x: 0,
      origin_z: 0,
      size: 1024,
      resolution: 256,
      max_elevation: 200,
      heightmap_b64: "", // not needed for this logic test
      biome_map_b64: "",
      encoding: "base64_uint16_uint8",
      survivor_count: 3,
    };

    // Replicate the logic from terrain.ts buildChunk
    const fullSize = chunk.size;
    const dataRes = chunk.resolution ?? fullSize;
    const renderSize =
      dataRes < fullSize
        ? dataRes
        : Math.floor(fullSize / 4); // CHUNK_DOWNSAMPLE_FALLBACK = 4
    const ds = dataRes < fullSize ? 1 : 4;
    const cellSize = fullSize / renderSize;

    // With resolution=256 and size=1024:
    //   dataRes=256, renderSize=256, ds=1, cellSize=4
    expect(dataRes).toBe(256);
    expect(renderSize).toBe(256);
    expect(ds).toBe(1);
    expect(cellSize).toBe(4); // 1024 / 256

    // Verify array index calculation: vertex (100, 50) should map to data index 50*256+100
    const col = 100;
    const row = 50;
    const srcCol = Math.min(col * ds, dataRes - 1);
    const srcRow = Math.min(row * ds, dataRes - 1);
    const srcIdx = srcRow * dataRes + srcCol;

    expect(srcCol).toBe(100); // ds=1, so 1:1 mapping
    expect(srcRow).toBe(50);
    expect(srcIdx).toBe(50 * 256 + 100);
  });

  it("falls back to downsample when resolution is absent (legacy)", () => {
    const chunk: ChunkTerrainData = {
      type: "chunk_terrain",
      cx: 0,
      cz: 0,
      origin_x: 0,
      origin_z: 0,
      size: 1024,
      // resolution is undefined (legacy data)
      max_elevation: 200,
      heightmap_b64: "",
      biome_map_b64: "",
      encoding: "base64_uint16_uint8",
      survivor_count: 0,
    };

    const fullSize = chunk.size;
    const dataRes = chunk.resolution ?? fullSize;
    const renderSize =
      dataRes < fullSize ? dataRes : Math.floor(fullSize / 4);
    const ds = dataRes < fullSize ? 1 : 4;

    // Without resolution field: dataRes=1024, must downsample client-side
    expect(dataRes).toBe(1024);
    expect(renderSize).toBe(256); // 1024 / 4
    expect(ds).toBe(4);
  });

  it("resolution field prevents double-downsample", () => {
    // This is the critical regression: if backend already downsampled to 256,
    // the frontend should NOT downsample again by 4x to 64.
    const chunk: ChunkTerrainData = {
      type: "chunk_terrain",
      cx: 1,
      cz: 2,
      origin_x: 1024,
      origin_z: 2048,
      size: 1024,
      resolution: 256,
      max_elevation: 200,
      heightmap_b64: "",
      biome_map_b64: "",
      encoding: "base64_uint16_uint8",
      survivor_count: 1,
    };

    const fullSize = chunk.size;
    const dataRes = chunk.resolution ?? fullSize;
    const renderSize =
      dataRes < fullSize ? dataRes : Math.floor(fullSize / 4);

    // renderSize MUST be 256 (from resolution), NOT 64 (from 256/4)
    expect(renderSize).toBe(256);
    expect(renderSize).not.toBe(64);
  });

  it("heightmap array size matches resolution squared", () => {
    // Backend sends uint16 heightmap: 256*256*2 bytes = 131072 bytes
    const resolution = 256;
    const expectedUint16Count = resolution * resolution;
    const expectedByteSize = expectedUint16Count * 2;

    expect(expectedByteSize).toBe(131072);

    // A 1024x1024 full-res heightmap would be 1024*1024*2 = 2097152 bytes
    const fullResBytes = 1024 * 1024 * 2;
    expect(fullResBytes).toBe(2097152);

    // The ratio shows why downsampling matters: 16x smaller
    expect(fullResBytes / expectedByteSize).toBe(16);
  });
});

// ---------------------------------------------------------------------------
// 2. FogOfWarRenderer initialization tracking
// ---------------------------------------------------------------------------

describe("FogOfWarRenderer isInitialized", () => {
  // We cannot import the real FogOfWarRenderer because it depends on Three.js
  // WebGL context. Instead, test the pattern that the renderer uses: tracking
  // a mesh reference as the initialization flag.

  class MockFogOfWarRenderer {
    private mesh: object | null = null;
    private pixelData: Uint8Array | null = null;

    isInitialized(): boolean {
      return this.mesh !== null;
    }

    initialize(worldWidth: number, worldHeight: number, gridWidth?: number, gridHeight?: number): void {
      this.dispose();
      const texW = gridWidth ?? worldWidth;
      const texH = gridHeight ?? worldHeight;
      this.pixelData = new Uint8Array(texW * texH * 4);
      // Fill with "unexplored"
      for (let i = 0; i < texW * texH; i++) {
        const base = i * 4;
        this.pixelData[base] = 0;
        this.pixelData[base + 1] = 0;
        this.pixelData[base + 2] = 0;
        this.pixelData[base + 3] = 217;
      }
      this.mesh = {}; // Simulate mesh creation
    }

    dispose(): void {
      this.mesh = null;
      this.pixelData = null;
    }

    getPixelData(): Uint8Array | null {
      return this.pixelData;
    }
  }

  let renderer: MockFogOfWarRenderer;

  beforeEach(() => {
    renderer = new MockFogOfWarRenderer();
  });

  it("returns false before init", () => {
    expect(renderer.isInitialized()).toBe(false);
  });

  it("returns true after init", () => {
    renderer.initialize(1024, 1024, 256, 256);
    expect(renderer.isInitialized()).toBe(true);
  });

  it("returns false after dispose", () => {
    renderer.initialize(1024, 1024);
    expect(renderer.isInitialized()).toBe(true);
    renderer.dispose();
    expect(renderer.isInitialized()).toBe(false);
  });

  it("can be re-initialized safely", () => {
    renderer.initialize(512, 512);
    expect(renderer.isInitialized()).toBe(true);
    renderer.initialize(1024, 1024);
    expect(renderer.isInitialized()).toBe(true);
  });

  it("allocates correct pixel buffer size", () => {
    renderer.initialize(1024, 1024, 256, 256);
    const data = renderer.getPixelData();
    expect(data).not.toBeNull();
    // 256 * 256 * 4 (RGBA) = 262144
    expect(data!.length).toBe(256 * 256 * 4);
  });

  it("initializes pixels as unexplored (opaque black)", () => {
    renderer.initialize(64, 64, 8, 8);
    const data = renderer.getPixelData()!;
    // Check first pixel: R=0, G=0, B=0, A=217
    expect(data[0]).toBe(0);
    expect(data[1]).toBe(0);
    expect(data[2]).toBe(0);
    expect(data[3]).toBe(217);
  });
});
