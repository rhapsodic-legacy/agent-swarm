import * as THREE from "three";
import type { DecodedTerrain } from "@/network/types";

/**
 * Biome color palette. Each entry is an [r, g, b] tuple in 0–1 range.
 * Order matches the biome enum: WATER=0, BEACH=1, FOREST=2, URBAN=3, MOUNTAIN=4, SNOW=5.
 */
const BIOME_COLORS: [number, number, number][] = [
  [0x1a / 255, 0x52 / 255, 0x76 / 255], // WATER  #1a5276
  [0xf0 / 255, 0xd9 / 255, 0xb5 / 255], // BEACH  #f0d9b5
  [0x2d / 255, 0x5a / 255, 0x27 / 255], // FOREST #2d5a27
  [0x80 / 255, 0x80 / 255, 0x80 / 255], // URBAN  #808080
  [0x6b / 255, 0x42 / 255, 0x26 / 255], // MOUNTAIN #6b4226
  [0xf0 / 255, 0xf0 / 255, 0xf0 / 255], // SNOW   #f0f0f0
];

/** Water level as a fraction of max_elevation. */
const WATER_LEVEL_FRACTION = 0.15;

/**
 * Simple deterministic pseudo-random number generator seeded by vertex index.
 * Returns a value in [-1, 1].
 */
function seededRandom(seed: number): number {
  // Fast integer hash (based on a common bit-mixing approach)
  let h = (seed * 2654435761) >>> 0;
  h = ((h ^ (h >>> 16)) * 0x45d9f3b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return (h / 0xffffffff) * 2 - 1;
}

/**
 * Apply ±5% deterministic variation to a color channel.
 * The result is clamped to [0, 1].
 */
function varyChannel(base: number, vertexIndex: number, channelOffset: number): number {
  const noise = seededRandom(vertexIndex * 3 + channelOffset) * 0.05;
  return Math.min(1, Math.max(0, base * (1 + noise)));
}

/**
 * Renders terrain as a displaced plane with vertex colors, plus a transparent
 * water plane. Designed for a drone swarm search-and-rescue simulation.
 *
 * Grid mapping: `grid[row][col]` -> world position `(col, heightmap[row][col], row)`.
 * The terrain occupies x=[0..width], z=[0..height].
 */
export class TerrainRenderer {
  private readonly scene: THREE.Scene;

  private terrainMesh: THREE.Mesh | null = null;
  private waterMesh: THREE.Mesh | null = null;
  private terrainGeometry: THREE.BufferGeometry | null = null;
  private terrainMaterial: THREE.MeshStandardMaterial | null = null;
  private waterGeometry: THREE.PlaneGeometry | null = null;
  private waterMaterial: THREE.MeshStandardMaterial | null = null;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /**
   * Build and add the terrain + water meshes from the provided data.
   * Should be called once when the initial terrain payload arrives over WebSocket.
   * Calling again will dispose the previous meshes first.
   */
  buildFromData(terrain: DecodedTerrain): void {
    // Clean up any previous build
    this.dispose();

    const { width, height, max_elevation, heightmap, biome_map } = terrain;

    // ---- Terrain mesh ---------------------------------------------------

    // PlaneGeometry creates a mesh in the XY plane. We rotate it to XZ later.
    // Segments = grid cells - 1, so vertex count = width * height.
    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);

    // Rotate from XY to XZ (face up). Rotation is -90 degrees around X.
    geometry.rotateX(-Math.PI / 2);

    const posAttr = geometry.getAttribute("position");
    const vertexCount = posAttr.count;

    // Prepare vertex color buffer
    const colors = new Float32Array(vertexCount * 3);

    // PlaneGeometry after -90° X rotation produces vertices ordered row-major,
    // top-to-bottom (decreasing Z) then left-to-right (increasing X).
    // The default plane is centered at origin with the given width/height.
    // We need to shift it so that grid[0][0] maps to world (0, elev, 0).
    //
    // Default vertex layout after rotation:
    //   x in [-width/2 .. +width/2]
    //   z in [-height/2 .. +height/2]   (but row 0 is at z = +height/2, descending)
    //
    // We want:
    //   grid[row][col] -> (col, elevation, row)
    //   col in [0 .. width-1]  mapped from width vertices across X
    //   row in [0 .. height-1] mapped from height vertices across Z
    //
    // Strategy: iterate vertices, compute their grid row/col from the default
    // layout, then overwrite positions directly.

    for (let i = 0; i < vertexCount; i++) {
      // PlaneGeometry is (width) segments across X, (height) segments across Z (after rotation).
      // Vertices are laid out row-by-row: row index along Z, column index along X.
      const col = i % width;
      const row = Math.floor(i / width);

      // Look up height and biome from flat typed arrays
      const idx = row * width + col;
      const elevation = heightmap[idx];
      const biome = biome_map[idx];

      // Set position: (col, elevation, row)
      posAttr.setXYZ(i, col, elevation, row);

      // Vertex color from biome with deterministic noise
      const biomeIndex = Math.min(biome, BIOME_COLORS.length - 1);
      const [br, bg, bb] = BIOME_COLORS[biomeIndex];
      colors[i * 3] = varyChannel(br, i, 0);
      colors[i * 3 + 1] = varyChannel(bg, i, 1);
      colors[i * 3 + 2] = varyChannel(bb, i, 2);
    }

    posAttr.needsUpdate = true;

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.0,
      flatShading: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;

    this.terrainGeometry = geometry;
    this.terrainMaterial = material;
    this.terrainMesh = mesh;
    this.scene.add(mesh);

    // ---- Water plane ----------------------------------------------------

    const waterLevel = max_elevation * WATER_LEVEL_FRACTION;

    const waterGeo = new THREE.PlaneGeometry(width, height);
    waterGeo.rotateX(-Math.PI / 2);

    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x1a6e8e,
      transparent: true,
      opacity: 0.6,
      roughness: 0.1,
      metalness: 0.3,
    });

    const water = new THREE.Mesh(waterGeo, waterMat);
    water.position.set(width / 2, waterLevel, height / 2);
    water.receiveShadow = true;

    this.waterGeometry = waterGeo;
    this.waterMaterial = waterMat;
    this.waterMesh = water;
    this.scene.add(water);
  }

  /**
   * Remove meshes from the scene and release all GPU resources.
   */
  dispose(): void {
    if (this.terrainMesh) {
      this.scene.remove(this.terrainMesh);
      this.terrainMesh = null;
    }
    if (this.waterMesh) {
      this.scene.remove(this.waterMesh);
      this.waterMesh = null;
    }

    this.terrainGeometry?.dispose();
    this.terrainGeometry = null;

    this.terrainMaterial?.dispose();
    this.terrainMaterial = null;

    this.waterGeometry?.dispose();
    this.waterGeometry = null;

    this.waterMaterial?.dispose();
    this.waterMaterial = null;
  }
}
