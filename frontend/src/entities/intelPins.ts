/**
 * LiveIntelPinRenderer — renders LLM / operator-placed priority pins as
 * pulsing ground circles with a vertical beacon beam, so the operator
 * can visually trace where they (or Claude) dropped a hint.
 *
 * Driven by `state.intel_pins` (backend is source of truth). TTL-expiring
 * pins disappear automatically when the backend drops them.
 */

import * as THREE from "three";
import type { LiveIntelPinData } from "@/network/types";

const PIN_COLOR = 0x00e5ff;        // cyan — distinct from zone greens / avoid reds
const PIN_OPACITY = 0.55;
const BEAM_OPACITY = 0.35;
const PULSE_SPEED = 2.2;
const PULSE_AMPLITUDE = 0.15;
const BEAM_HEIGHT = 240;           // how far up the beacon rises

interface PinVisual {
  group: THREE.Group;
  ring: THREE.Mesh;
  beam: THREE.Mesh;
  fingerprint: string;
}

function fingerprintPin(p: LiveIntelPinData): string {
  return `${p.pin_id}|${p.position[0].toFixed(1)},${p.position[1].toFixed(1)}|${p.radius.toFixed(0)}`;
}

export class LiveIntelPinRenderer {
  private scene: THREE.Scene;
  private visuals: Map<string, PinVisual> = new Map();
  private pulseTime = 0;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  update(pins: LiveIntelPinData[] | undefined, dt: number): void {
    const current = pins ?? [];
    const seen = new Set<string>();

    for (const pin of current) {
      seen.add(pin.pin_id);
      const fp = fingerprintPin(pin);
      const existing = this.visuals.get(pin.pin_id);
      if (existing && existing.fingerprint === fp) continue;
      if (existing) this.removeVisual(pin.pin_id);
      this.addVisual(pin, fp);
    }

    for (const id of Array.from(this.visuals.keys())) {
      if (!seen.has(id)) this.removeVisual(id);
    }

    this.pulseTime += dt * PULSE_SPEED;
    const pulse = 1.0 + Math.sin(this.pulseTime) * PULSE_AMPLITUDE;
    for (const v of this.visuals.values()) {
      v.ring.scale.set(pulse, 1.0, pulse);
      const beamMat = v.beam.material as THREE.MeshBasicMaterial;
      beamMat.opacity = BEAM_OPACITY * (0.7 + 0.3 * Math.sin(this.pulseTime * 0.8));
    }
  }

  private addVisual(pin: LiveIntelPinData, fingerprint: string): void {
    const group = new THREE.Group();
    group.position.set(pin.position[0], 0, pin.position[1]);

    // Ground ring — a flat disk pulsing on the terrain
    const ringGeo = new THREE.RingGeometry(
      Math.max(pin.radius - 20, 10),
      pin.radius,
      48,
    );
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: PIN_COLOR,
      transparent: true,
      opacity: PIN_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.y = 2.0;
    ring.renderOrder = 1200;
    group.add(ring);

    // Vertical beacon beam — a narrow translucent cylinder
    const beamGeo = new THREE.CylinderGeometry(6, 6, BEAM_HEIGHT, 16, 1, true);
    const beamMat = new THREE.MeshBasicMaterial({
      color: PIN_COLOR,
      transparent: true,
      opacity: BEAM_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false,
    });
    const beam = new THREE.Mesh(beamGeo, beamMat);
    beam.position.y = BEAM_HEIGHT / 2;
    beam.renderOrder = 1201;
    group.add(beam);

    this.scene.add(group);
    this.visuals.set(pin.pin_id, { group, ring, beam, fingerprint });
  }

  private removeVisual(pinId: string): void {
    const v = this.visuals.get(pinId);
    if (!v) return;
    this.scene.remove(v.group);
    v.ring.geometry.dispose();
    (v.ring.material as THREE.Material).dispose();
    v.beam.geometry.dispose();
    (v.beam.material as THREE.Material).dispose();
    this.visuals.delete(pinId);
  }

  dispose(): void {
    for (const id of Array.from(this.visuals.keys())) this.removeVisual(id);
  }
}
