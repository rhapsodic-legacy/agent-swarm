import * as THREE from "three";
import { FogData } from "@/network/types";

/**
 * RGBA values for each fog state, packed as [R, G, B, A] bytes.
 *
 *   0 = unexplored  → black, high opacity
 *   1 = explored     → fully transparent
 *   2 = stale        → black, slight tint
 */
const FOG_RGBA: Record<number, [number, number, number, number]> = {
  0: [0, 0, 0, 217], // 0.85 * 255 ≈ 217
  1: [0, 0, 0, 0],
  2: [0, 0, 0, 77], // 0.30 * 255 ≈ 77
};

/** Fallback for any unexpected RLE value — treat as unexplored. */
const FALLBACK_RGBA: [number, number, number, number] = FOG_RGBA[0];

/** Fixed Y position of the fog overlay plane (above max terrain elevation of 200). */
const FOG_PLANE_Y = 210;

/** Render order for the fog mesh — ensures it draws on top of other scene objects. */
const FOG_RENDER_ORDER = 10;

export class FogOfWarRenderer {
  private readonly scene: THREE.Scene;
  private texture: THREE.DataTexture | null = null;
  private material: THREE.MeshBasicMaterial | null = null;
  private geometry: THREE.PlaneGeometry | null = null;
  private mesh: THREE.Mesh | null = null;
  private pixelData: Uint8Array | null = null;
  private _scratchValues: Uint8Array | null = null;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /**
   * Set up the fog overlay plane. Call once when terrain/world dimensions are known.
   *
   * @param worldWidth  World width in meters (used for plane geometry size + position).
   * @param worldHeight World height in meters.
   * @param gridWidth   Fog grid resolution width (texture pixels). Defaults to worldWidth.
   * @param gridHeight  Fog grid resolution height (texture pixels). Defaults to worldHeight.
   */
  initialize(worldWidth: number, worldHeight: number, gridWidth?: number, gridHeight?: number): void {
    // Tear down any previous state so `initialize` is safely re-callable.
    this.dispose();

    const texW = gridWidth ?? worldWidth;
    const texH = gridHeight ?? worldHeight;

    // Allocate the shared pixel buffer (RGBA, one byte per channel).
    this.pixelData = new Uint8Array(texW * texH * 4);

    // Fill with "unexplored" by default.
    const [r, g, b, a] = FOG_RGBA[0];
    for (let i = 0; i < texW * texH; i++) {
      const base = i * 4;
      this.pixelData[base] = r;
      this.pixelData[base + 1] = g;
      this.pixelData[base + 2] = b;
      this.pixelData[base + 3] = a;
    }

    // Create the DataTexture.
    this.texture = new THREE.DataTexture(
      this.pixelData as unknown as BufferSource,
      texW,
      texH,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.flipY = false; // Row 0 = Z=0 (south), maps to bottom of UV — no flip needed
    this.texture.needsUpdate = true;

    // Build the plane geometry sized to the WORLD, rotated so it lies in the XZ plane (Y-up).
    this.geometry = new THREE.PlaneGeometry(worldWidth, worldHeight);
    this.geometry.rotateX(-Math.PI / 2);

    // Transparent material with no depth-write so it doesn't occlude objects.
    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.position.set(worldWidth / 2, FOG_PLANE_Y, worldHeight / 2);
    this.mesh.renderOrder = FOG_RENDER_ORDER;

    this.scene.add(this.mesh);
  }

  /** Whether the fog overlay has been initialized with dimensions. */
  isInitialized(): boolean {
    return this.mesh !== null;
  }

  /**
   * Decode RLE fog data and update the texture in-place.
   *
   * The fog grid is row-major with row 0 = world Z=0 (south). But after
   * PlaneGeometry.rotateX(-PI/2), UV v=0 maps to world Z=max (north).
   * So we must flip rows: fog row 0 → texture row (height-1), etc.
   */
  updateFromRLE(fogData: FogData): void {
    const data = this.pixelData;
    if (!data || !this.texture) {
      return;
    }

    const { width, height, rle } = fogData;
    // Decode RLE into a flat value array (one byte per cell)
    const totalPixels = width * height;
    // Reuse a scratch buffer to avoid allocation every frame
    if (!this._scratchValues || this._scratchValues.length < totalPixels) {
      this._scratchValues = new Uint8Array(totalPixels);
    }
    const values = this._scratchValues;
    let idx = 0;
    for (let i = 0; i < rle.length; i++) {
      const [value, count] = rle[i];
      const end = idx + count;
      for (let p = idx; p < end; p++) {
        values[p] = value;
      }
      idx = end;
    }

    // Write to pixelData with row-flip: fog row r → texture row (height - 1 - r)
    for (let fogRow = 0; fogRow < height; fogRow++) {
      const texRow = height - 1 - fogRow;
      const fogRowStart = fogRow * width;
      const texRowStart = texRow * width;
      for (let col = 0; col < width; col++) {
        const value = values[fogRowStart + col];
        const rgba = FOG_RGBA[value] ?? FALLBACK_RGBA;
        const base = (texRowStart + col) * 4;
        data[base] = rgba[0];
        data[base + 1] = rgba[1];
        data[base + 2] = rgba[2];
        data[base + 3] = rgba[3];
      }
    }

    this.texture.needsUpdate = true;
  }

  /** Show or hide the fog overlay without disposing it. */
  setVisible(visible: boolean): void {
    if (this.mesh) {
      this.mesh.visible = visible;
    }
  }

  /** Remove the overlay from the scene and release all GPU resources. */
  dispose(): void {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh = null;
    }
    if (this.geometry) {
      this.geometry.dispose();
      this.geometry = null;
    }
    if (this.material) {
      this.material.dispose();
      this.material = null;
    }
    if (this.texture) {
      this.texture.dispose();
      this.texture = null;
    }
    this.pixelData = null;
    this._scratchValues = null;
  }
}
