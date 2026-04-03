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

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /**
   * Set up the fog overlay plane. Call once when terrain dimensions are known.
   *
   * @param width  Grid width in cells (meters).
   * @param height Grid height in cells (meters).
   */
  initialize(width: number, height: number): void {
    // Tear down any previous state so `initialize` is safely re-callable.
    this.dispose();

    // Allocate the shared pixel buffer (RGBA, one byte per channel).
    this.pixelData = new Uint8Array(width * height * 4);

    // Fill with "unexplored" by default.
    const [r, g, b, a] = FOG_RGBA[0];
    for (let i = 0; i < width * height; i++) {
      const base = i * 4;
      this.pixelData[base] = r;
      this.pixelData[base + 1] = g;
      this.pixelData[base + 2] = b;
      this.pixelData[base + 3] = a;
    }

    // Create the DataTexture.
    this.texture = new THREE.DataTexture(
      this.pixelData as unknown as BufferSource,
      width,
      height,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.needsUpdate = true;

    // Build the plane geometry, rotated so it lies in the XZ plane (Y-up).
    this.geometry = new THREE.PlaneGeometry(width, height);
    this.geometry.rotateX(-Math.PI / 2);

    // Transparent material with no depth-write so it doesn't occlude objects.
    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.position.set(width / 2, FOG_PLANE_Y, height / 2);
    this.mesh.renderOrder = FOG_RENDER_ORDER;

    this.scene.add(this.mesh);
  }

  /**
   * Decode RLE fog data and update the texture in-place.
   *
   * Each `[value, count]` pair in `fogData.rle` writes `count` consecutive
   * pixels (row-major order) with the RGBA colour for that fog state.
   */
  updateFromRLE(fogData: FogData): void {
    const data = this.pixelData;
    if (!data || !this.texture) {
      return;
    }

    const { rle } = fogData;
    let pixelIndex = 0;

    for (let i = 0; i < rle.length; i++) {
      const [value, count] = rle[i];
      const rgba = FOG_RGBA[value] ?? FALLBACK_RGBA;
      const [r, g, b, a] = rgba;

      const end = pixelIndex + count;
      for (let p = pixelIndex; p < end; p++) {
        const base = p * 4;
        data[base] = r;
        data[base + 1] = g;
        data[base + 2] = b;
        data[base + 3] = a;
      }

      pixelIndex = end;
    }

    // Signal the GPU to re-upload the texture data exactly once per update.
    this.texture.needsUpdate = true;
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
  }
}
