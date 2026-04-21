/**
 * ZoneRenderer — renders operator-drawn priority zones as colored rectangles
 * on the ground plane. Each zone has a translucent fill + crisp border so
 * it's legible from both ground and overview camera angles.
 *
 * Driven by `state.zones` (backend is source of truth); input capture lives
 * in ZoneTool. We rebuild meshes only when the zone set actually changes —
 * identified by a cheap id + priority + polygon fingerprint.
 */

import * as THREE from "three";
import type { ZoneData } from "@/network/types";

const PRIORITY_COLORS: Record<string, number> = {
  high: 0x44ff88,
  low: 0xffcc44,
  avoid: 0xff4466,
};

const FILL_OPACITY = 0.15;
const BORDER_OPACITY = 0.85;
const PULSE_SPEED = 1.5;
const PULSE_AMPLITUDE = 0.04;

interface ZoneVisual {
  fill: THREE.Mesh;
  border: THREE.Line;
  fingerprint: string;
}

function fingerprintZone(z: ZoneData): string {
  return `${z.zone_id}|${z.priority}|${z.polygon.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(";")}`;
}

export class ZoneRenderer {
  private scene: THREE.Scene;
  private visuals: Map<string, ZoneVisual> = new Map();
  private pulseTime = 0;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  update(zones: ZoneData[] | undefined, dt: number): void {
    const current = zones ?? [];
    const seen = new Set<string>();

    for (const zone of current) {
      seen.add(zone.zone_id);
      const fp = fingerprintZone(zone);
      const existing = this.visuals.get(zone.zone_id);
      if (existing && existing.fingerprint === fp) continue;
      if (existing) this.removeVisual(zone.zone_id);
      this.addVisual(zone, fp);
    }

    // Remove zones that no longer exist
    for (const zoneId of Array.from(this.visuals.keys())) {
      if (!seen.has(zoneId)) this.removeVisual(zoneId);
    }

    // Gentle pulse to make zones feel "active" without being noisy
    this.pulseTime += dt * PULSE_SPEED;
    const pulse = 1.0 + Math.sin(this.pulseTime) * PULSE_AMPLITUDE;
    for (const v of this.visuals.values()) {
      const mat = v.fill.material as THREE.MeshBasicMaterial;
      mat.opacity = FILL_OPACITY * pulse;
    }
  }

  private addVisual(zone: ZoneData, fingerprint: string): void {
    const color = PRIORITY_COLORS[zone.priority] ?? 0xffffff;
    const polygon = zone.polygon;
    if (polygon.length < 3) return;

    // Fill: ShapeGeometry over the polygon projected to XZ.
    const shape = new THREE.Shape(
      polygon.map(([x, z]) => new THREE.Vector2(x, z)),
    );
    const fillGeo = new THREE.ShapeGeometry(shape);
    // Rotate so the shape lies on the XZ plane (ShapeGeometry is in XY).
    fillGeo.rotateX(-Math.PI / 2);
    // Move slightly above ground to avoid z-fighting with terrain.
    fillGeo.translate(0, 0.8, 0);

    const fillMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: FILL_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const fill = new THREE.Mesh(fillGeo, fillMat);
    fill.renderOrder = 5;

    // Border: closed line loop at a fractionally higher Y.
    const borderPositions = new Float32Array((polygon.length + 1) * 3);
    for (let i = 0; i < polygon.length; i++) {
      borderPositions[i * 3] = polygon[i][0];
      borderPositions[i * 3 + 1] = 1.0;
      borderPositions[i * 3 + 2] = polygon[i][1];
    }
    borderPositions[polygon.length * 3] = polygon[0][0];
    borderPositions[polygon.length * 3 + 1] = 1.0;
    borderPositions[polygon.length * 3 + 2] = polygon[0][1];
    const borderGeo = new THREE.BufferGeometry();
    borderGeo.setAttribute(
      "position",
      new THREE.BufferAttribute(borderPositions, 3),
    );
    const borderMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: BORDER_OPACITY,
    });
    const border = new THREE.Line(borderGeo, borderMat);
    border.renderOrder = 6;

    this.scene.add(fill);
    this.scene.add(border);
    this.visuals.set(zone.zone_id, { fill, border, fingerprint });
  }

  private removeVisual(zoneId: string): void {
    const v = this.visuals.get(zoneId);
    if (!v) return;
    this.scene.remove(v.fill);
    v.fill.geometry.dispose();
    (v.fill.material as THREE.Material).dispose();
    this.scene.remove(v.border);
    v.border.geometry.dispose();
    (v.border.material as THREE.Material).dispose();
    this.visuals.delete(zoneId);
  }

  dispose(): void {
    for (const id of Array.from(this.visuals.keys())) this.removeVisual(id);
  }
}
