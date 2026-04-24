/**
 * WindFieldRenderer — a ground-plane grid of arrows showing the current
 * global wind direction and magnitude. Instanced for cheap draw cost.
 *
 * The backend broadcasts a single global wind vector per tick (see
 * WeatherInfo), so every arrow points the same way; the grid exists to
 * make the field legible across a 10km world and to give the scene a
 * sense of environmental motion.
 */

import * as THREE from "three";

import type { WeatherInfo } from "@/network/types";

const GRID_CELLS = 12;               // GRID_CELLS × GRID_CELLS arrows
const ARROW_LENGTH = 80;             // meters; scaled with wind speed
const ARROW_WIDTH = 20;
const ARROW_Y = 5;                   // float above ground so terrain doesn't z-fight
const CALM_COLOR = new THREE.Color(0x66aaff);
const GUST_COLOR = new THREE.Color(0xff5533);
const GUST_SPEED_MS = 8.0;           // above this we interpolate toward GUST_COLOR
const BASE_OPACITY = 0.55;

export class WindFieldRenderer {
  private scene: THREE.Scene;
  private mesh: THREE.InstancedMesh | null = null;
  private material: THREE.MeshBasicMaterial | null = null;
  private worldSize = 0;
  private readonly instanceCount = GRID_CELLS * GRID_CELLS;

  // Scratch
  private readonly mat4 = new THREE.Matrix4();
  private readonly pos = new THREE.Vector3();
  private readonly quat = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3();
  private readonly axisY = new THREE.Vector3(0, 1, 0);

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /** Initialize once the world size is known. Idempotent. */
  initialize(worldSize: number): void {
    if (this.mesh) return;
    this.worldSize = worldSize;

    // Arrow shape: a flat triangle in the XZ plane pointing along +X.
    // Built from a BufferGeometry so we can lay it flat and instance it.
    const shape = new THREE.Shape();
    const tail = -ARROW_LENGTH / 2;
    const tip = ARROW_LENGTH / 2;
    const halfW = ARROW_WIDTH / 2;
    shape.moveTo(tip, 0);
    shape.lineTo(tip - ARROW_WIDTH, halfW);
    shape.lineTo(tip - ARROW_WIDTH, halfW * 0.5);
    shape.lineTo(tail, halfW * 0.3);
    shape.lineTo(tail, -halfW * 0.3);
    shape.lineTo(tip - ARROW_WIDTH, -halfW * 0.5);
    shape.lineTo(tip - ARROW_WIDTH, -halfW);
    shape.lineTo(tip, 0);

    const geo = new THREE.ShapeGeometry(shape);
    // ShapeGeometry lives in XY; rotate so the arrow lies flat on the XZ plane.
    geo.rotateX(-Math.PI / 2);

    this.material = new THREE.MeshBasicMaterial({
      color: CALM_COLOR,
      transparent: true,
      opacity: BASE_OPACITY,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.InstancedMesh(geo, this.material, this.instanceCount);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;  // draw above terrain but below UI overlays
    this.scene.add(this.mesh);

    // Seed base positions (they don't change — only rotation + scale + color).
    const cellSize = worldSize / GRID_CELLS;
    const offset = cellSize / 2;
    let i = 0;
    for (let gz = 0; gz < GRID_CELLS; gz++) {
      for (let gx = 0; gx < GRID_CELLS; gx++) {
        this.pos.set(gx * cellSize + offset, ARROW_Y, gz * cellSize + offset);
        this.quat.identity();
        this.scale.set(1, 1, 1);
        this.mat4.compose(this.pos, this.quat, this.scale);
        this.mesh.setMatrixAt(i, this.mat4);
        i++;
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  /** Update rotation + scale from current weather. No-op if not initialized. */
  update(weather: WeatherInfo | undefined): void {
    if (!this.mesh || !weather) return;

    // wind_direction is the angle in world XZ. Our arrow points along +X in
    // local space. A rotation about +Y by `-dir` (right-hand rule) maps +X
    // to the bearing (cos dir, 0, sin dir) that matches get_wind_at.
    // Empirically: get_wind_at returns (cos d, 0, sin d) × speed — so
    // rotate by `-wind_direction` around Y.
    this.quat.setFromAxisAngle(this.axisY, -weather.wind_direction);

    // Scale arrow length with speed (clamped so it never disappears or
    // swamps the scene).
    const speedScale = Math.max(0.35, Math.min(1.6, weather.wind_speed / 6.0));
    this.scale.set(speedScale, 1, speedScale);

    const cellSize = this.worldSize / GRID_CELLS;
    const offset = cellSize / 2;

    let i = 0;
    for (let gz = 0; gz < GRID_CELLS; gz++) {
      for (let gx = 0; gx < GRID_CELLS; gx++) {
        this.pos.set(gx * cellSize + offset, ARROW_Y, gz * cellSize + offset);
        this.mat4.compose(this.pos, this.quat, this.scale);
        this.mesh.setMatrixAt(i, this.mat4);
        i++;
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;

    // Color: interpolate calm → gust based on wind speed.
    if (this.material) {
      const t = Math.min(1.0, weather.wind_speed / GUST_SPEED_MS);
      this.material.color.copy(CALM_COLOR).lerp(GUST_COLOR, t);
      this.material.opacity = BASE_OPACITY + (weather.gusting ? 0.2 : 0.0);
    }
  }

  dispose(): void {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh = null;
    }
    if (this.material) {
      this.material.dispose();
      this.material = null;
    }
  }
}
