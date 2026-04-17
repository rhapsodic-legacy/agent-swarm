/**
 * PoC (Probability of Containment) heatmap overlay.
 *
 * Renders the Bayesian search map as a color-coded plane floating above the
 * terrain. Red = high probability, blue = low, transparent = near-zero.
 *
 * Toggle with the H key. Updates whenever a new poc_grid arrives (1 Hz).
 */

import * as THREE from "three";
import type { PoCGrid } from "@/network/types";

const HEATMAP_ALTITUDE = 220; // Above max terrain elevation (200)
const RENDER_ORDER = 9;

// Colormap: blue → cyan → green → yellow → red (classic "jet")
function valueToColor(v: number): [number, number, number, number] {
  // v in [0, 1]
  const clamped = Math.max(0, Math.min(1, v));
  let r: number, g: number, b: number;
  if (clamped < 0.25) {
    // blue → cyan
    const t = clamped / 0.25;
    r = 0;
    g = Math.round(255 * t);
    b = 255;
  } else if (clamped < 0.5) {
    // cyan → green
    const t = (clamped - 0.25) / 0.25;
    r = 0;
    g = 255;
    b = Math.round(255 * (1 - t));
  } else if (clamped < 0.75) {
    // green → yellow
    const t = (clamped - 0.5) / 0.25;
    r = Math.round(255 * t);
    g = 255;
    b = 0;
  } else {
    // yellow → red
    const t = (clamped - 0.75) / 0.25;
    r = 255;
    g = Math.round(255 * (1 - t));
    b = 0;
  }
  // Alpha scales with value — near-zero cells are nearly transparent
  const a = Math.round(200 * clamped);
  return [r, g, b, a];
}

export class PoCHeatmap {
  private scene: THREE.Scene;
  private mesh: THREE.Mesh | null = null;
  private texture: THREE.DataTexture | null = null;
  private pixels: Uint8Array | null = null;
  private currentSize: number = 0;
  private currentWorldSize: number = 0;
  private active = false;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  toggle(): void {
    this.active = !this.active;
    if (this.mesh) this.mesh.visible = this.active;
    console.log(`[PoCHeatmap] ${this.active ? "ON" : "OFF"}`);
  }

  isActive(): boolean {
    return this.active;
  }

  update(grid: PoCGrid | undefined): void {
    if (!grid) return;

    // Rebuild mesh if size changed (world reset)
    if (grid.size !== this.currentSize || grid.world_size !== this.currentWorldSize) {
      this.disposeMesh();
      this.createMesh(grid.size, grid.world_size);
    }

    if (!this.pixels || !this.texture) return;

    // Decode base64 → uint8 (cells 0-255)
    const bytes = Uint8Array.from(atob(grid.data_b64), (c) => c.charCodeAt(0));
    if (bytes.length !== grid.size * grid.size) {
      console.warn(
        `[PoCHeatmap] size mismatch: got ${bytes.length} bytes, expected ${grid.size * grid.size}`,
      );
      return;
    }

    // Map each cell to RGBA color. Texture is top-to-bottom, so flip rows
    // to match world-space (z increases south = down on the texture).
    const n = grid.size;
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < n; col++) {
        const v = bytes[row * n + col] / 255;
        const [r, g, b, a] = valueToColor(v);
        // Texture row flip: texture origin is bottom-left, world z grows away
        const texRow = n - 1 - row;
        const idx = (texRow * n + col) * 4;
        this.pixels[idx] = r;
        this.pixels[idx + 1] = g;
        this.pixels[idx + 2] = b;
        this.pixels[idx + 3] = a;
      }
    }
    this.texture.needsUpdate = true;
  }

  private createMesh(size: number, worldSize: number): void {
    this.currentSize = size;
    this.currentWorldSize = worldSize;
    this.pixels = new Uint8Array(size * size * 4);

    this.texture = new THREE.DataTexture(
      this.pixels as unknown as BufferSource,
      size, size,
      THREE.RGBAFormat, THREE.UnsignedByteType,
    );
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.needsUpdate = true;

    const geometry = new THREE.PlaneGeometry(worldSize, worldSize);
    geometry.rotateX(-Math.PI / 2);

    const material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false, // Preserve full-strength jet colormap colors
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.position.set(worldSize / 2, HEATMAP_ALTITUDE, worldSize / 2);
    this.mesh.renderOrder = RENDER_ORDER;
    this.mesh.visible = this.active;
    this.scene.add(this.mesh);
  }

  private disposeMesh(): void {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      (this.mesh.geometry as THREE.BufferGeometry).dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.mesh = null;
    }
    if (this.texture) {
      this.texture.dispose();
      this.texture = null;
    }
    this.pixels = null;
  }

  dispose(): void {
    this.disposeMesh();
  }
}
