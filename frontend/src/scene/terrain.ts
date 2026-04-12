import * as THREE from "three";
import type { DecodedTerrain, ChunkTerrainData } from "@/network/types";

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

/** Fallback downsample factor when backend sends full-res data (legacy). */
const CHUNK_DOWNSAMPLE_FALLBACK = 4;

/**
 * Simple deterministic pseudo-random number generator seeded by vertex index.
 */
function seededRandom(seed: number): number {
  let h = (seed * 2654435761) >>> 0;
  h = ((h ^ (h >>> 16)) * 0x45d9f3b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return (h / 0xffffffff) * 2 - 1;
}

function varyChannel(base: number, vertexIndex: number, channelOffset: number): number {
  const noise = seededRandom(vertexIndex * 3 + channelOffset) * 0.05;
  return Math.min(1, Math.max(0, base * (1 + noise)));
}

/** Data for a single loaded chunk mesh. */
interface ChunkMesh {
  terrainMesh: THREE.Mesh;
  waterMesh: THREE.Mesh;
  geometry: THREE.BufferGeometry;
  material: THREE.MeshStandardMaterial;
  waterGeometry: THREE.PlaneGeometry;
  waterMaterial: THREE.MeshStandardMaterial;
}

/**
 * Renders terrain as displaced planes with vertex colors.
 * Supports both monolithic terrain (legacy) and chunked world (new).
 */
export class TerrainRenderer {
  private readonly scene: THREE.Scene;

  // Legacy monolithic mesh (used when monolithic terrain arrives)
  private terrainMesh: THREE.Mesh | null = null;
  private waterMesh: THREE.Mesh | null = null;
  private terrainGeometry: THREE.BufferGeometry | null = null;
  private terrainMaterial: THREE.MeshStandardMaterial | null = null;
  private waterGeometry: THREE.PlaneGeometry | null = null;
  private waterMaterial: THREE.MeshStandardMaterial | null = null;

  // Chunked meshes — keyed by "cx,cz"
  private chunks: Map<string, ChunkMesh> = new Map();

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /**
   * Build monolithic terrain mesh (legacy path).
   */
  buildFromData(terrain: DecodedTerrain): void {
    this.dispose();

    const { width, height, max_elevation, heightmap, biome_map } = terrain;

    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);
    geometry.rotateX(-Math.PI / 2);

    const posAttr = geometry.getAttribute("position");
    const vertexCount = posAttr.count;
    const colors = new Float32Array(vertexCount * 3);

    for (let i = 0; i < vertexCount; i++) {
      const col = i % width;
      const row = Math.floor(i / width);
      const idx = row * width + col;
      const elevation = heightmap[idx];
      const biome = biome_map[idx];

      posAttr.setXYZ(i, col, elevation, row);

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

    // Water plane
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
   * Build a chunk mesh from chunk terrain data (chunked world path).
   * Downsamples 1024x1024 to 256x256 for rendering performance.
   */
  buildChunk(chunk: ChunkTerrainData): void {
    const key = `${chunk.cx},${chunk.cz}`;

    // Dispose existing chunk mesh if reloading
    const existing = this.chunks.get(key);
    if (existing) {
      this.disposeChunkMesh(existing);
      this.chunks.delete(key);
    }

    // Decode base64 terrain data
    const hmBytes = Uint8Array.from(atob(chunk.heightmap_b64), (c) => c.charCodeAt(0));
    const hmUint16 = new Uint16Array(hmBytes.buffer);
    const bmBytes = Uint8Array.from(atob(chunk.biome_map_b64), (c) => c.charCodeAt(0));

    const fullSize = chunk.size;
    // Backend may send pre-downsampled data with a resolution field
    const dataRes = chunk.resolution ?? fullSize;
    // If data is still full-res (legacy), downsample client-side
    const renderSize = dataRes < fullSize ? dataRes : Math.floor(fullSize / CHUNK_DOWNSAMPLE_FALLBACK);
    const ds = dataRes < fullSize ? 1 : CHUNK_DOWNSAMPLE_FALLBACK;
    const cellSize = fullSize / renderSize; // meters per render cell

    // Create geometry
    const geometry = new THREE.PlaneGeometry(fullSize, fullSize, renderSize - 1, renderSize - 1);
    geometry.rotateX(-Math.PI / 2);

    const posAttr = geometry.getAttribute("position");
    const vertexCount = posAttr.count;
    const colors = new Float32Array(vertexCount * 3);

    for (let i = 0; i < vertexCount; i++) {
      const col = i % renderSize;
      const row = Math.floor(i / renderSize);

      // Map render grid to data grid
      const srcCol = Math.min(col * ds, dataRes - 1);
      const srcRow = Math.min(row * ds, dataRes - 1);
      const srcIdx = srcRow * dataRes + srcCol;

      const elevation = (hmUint16[srcIdx] / 65535) * chunk.max_elevation;
      const biome = bmBytes[srcIdx];

      // Position relative to chunk origin
      const worldX = col * cellSize;
      const worldZ = row * cellSize;
      posAttr.setXYZ(i, worldX, elevation, worldZ);

      const biomeIndex = Math.min(biome, BIOME_COLORS.length - 1);
      const [br, bg, bb] = BIOME_COLORS[biomeIndex];
      colors[i * 3] = varyChannel(br, i + chunk.cx * 10000 + chunk.cz * 100, 0);
      colors[i * 3 + 1] = varyChannel(bg, i + chunk.cx * 10000 + chunk.cz * 100, 1);
      colors[i * 3 + 2] = varyChannel(bb, i + chunk.cx * 10000 + chunk.cz * 100, 2);
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

    const terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.receiveShadow = true;
    // Position mesh at chunk's world origin
    terrainMesh.position.set(chunk.origin_x, 0, chunk.origin_z);
    this.scene.add(terrainMesh);

    // Water plane for this chunk
    const waterLevel = chunk.max_elevation * WATER_LEVEL_FRACTION;
    const waterGeo = new THREE.PlaneGeometry(fullSize, fullSize);
    waterGeo.rotateX(-Math.PI / 2);

    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x1a6e8e,
      transparent: true,
      opacity: 0.6,
      roughness: 0.1,
      metalness: 0.3,
    });

    const waterMesh = new THREE.Mesh(waterGeo, waterMat);
    waterMesh.position.set(
      chunk.origin_x + fullSize / 2,
      waterLevel,
      chunk.origin_z + fullSize / 2,
    );
    waterMesh.receiveShadow = true;
    this.scene.add(waterMesh);

    this.chunks.set(key, {
      terrainMesh,
      waterMesh,
      geometry,
      material,
      waterGeometry: waterGeo,
      waterMaterial: waterMat,
    });
  }

  /** Get the number of loaded chunks. */
  getChunkCount(): number {
    return this.chunks.size;
  }

  /** Check if a chunk is loaded. */
  hasChunk(cx: number, cz: number): boolean {
    return this.chunks.has(`${cx},${cz}`);
  }

  private disposeChunkMesh(cm: ChunkMesh): void {
    this.scene.remove(cm.terrainMesh);
    this.scene.remove(cm.waterMesh);
    cm.geometry.dispose();
    cm.material.dispose();
    cm.waterGeometry.dispose();
    cm.waterMaterial.dispose();
  }

  /**
   * Remove meshes from the scene and release all GPU resources.
   */
  dispose(): void {
    // Dispose monolithic meshes
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

    // Dispose all chunk meshes
    for (const cm of this.chunks.values()) {
      this.disposeChunkMesh(cm);
    }
    this.chunks.clear();
  }
}
