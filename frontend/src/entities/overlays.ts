/**
 * Renders communication links between drones and survivor markers.
 */

import * as THREE from "three";
import type { DroneState, EvidenceMarker, SurvivorState } from "@/network/types";

const COMM_LINK_COLOR = new THREE.Color(0x00aaff);
const COMM_LINK_OPACITY = 0.15;
const SURVIVOR_COLOR = new THREE.Color(0xff2200);
const SURVIVOR_PULSE_SPEED = 3.0;
const SURVIVOR_SCALE = 8; // scale factor for visibility in large (10km) worlds

// Evidence markers — one material/geometry per kind so each reads clearly
// from the overview camera angle in a 10km world.
const EVIDENCE_COLORS: Record<string, number> = {
  footprint: 0xffb060, // warm amber — human trace
  debris: 0xcc3322,    // desaturated red — hard debris
  signal_fire: 0xffee33, // bright yellow — active signal
};
const EVIDENCE_SCALE = 18; // read distance from orbit camera
const EVIDENCE_PULSE_SPEED = 2.2;

export class OverlayRenderer {
  private scene: THREE.Scene;

  // Communication links
  private commLinesMesh: THREE.LineSegments | null = null;
  private commGeometry: THREE.BufferGeometry;
  private commMaterial: THREE.LineBasicMaterial;
  private commPositions: Float32Array;
  private maxLinks: number;

  // Survivor markers
  private survivorMesh: THREE.InstancedMesh;
  private survivorMaterial: THREE.MeshStandardMaterial;
  private survivorPulseTime = 0;
  private readonly maxSurvivors: number;

  // Evidence markers — one pool per kind.
  private evidenceMeshes: Map<string, THREE.InstancedMesh> = new Map();
  private evidenceMaterials: Map<string, THREE.MeshStandardMaterial> = new Map();
  private readonly maxEvidence = 64;
  private evidencePulseTime = 0;

  // Scratch objects
  private readonly mat4 = new THREE.Matrix4();
  private readonly quat = new THREE.Quaternion();
  private readonly scaleVec = new THREE.Vector3();
  private readonly posVec = new THREE.Vector3();

  constructor(scene: THREE.Scene, maxDrones: number, maxSurvivors: number) {
    this.scene = scene;
    this.maxSurvivors = maxSurvivors;

    // --- Comm links ---
    // Max links = n*(n-1)/2 for n drones, each link = 2 points = 6 floats
    this.maxLinks = (maxDrones * (maxDrones - 1)) / 2;
    this.commPositions = new Float32Array(this.maxLinks * 6);
    this.commGeometry = new THREE.BufferGeometry();
    this.commGeometry.setAttribute("position", new THREE.BufferAttribute(this.commPositions, 3));
    this.commGeometry.setDrawRange(0, 0);
    this.commMaterial = new THREE.LineBasicMaterial({
      color: COMM_LINK_COLOR,
      transparent: true,
      opacity: COMM_LINK_OPACITY,
      depthWrite: false,
    });
    this.commLinesMesh = new THREE.LineSegments(this.commGeometry, this.commMaterial);
    this.commLinesMesh.frustumCulled = false;
    this.scene.add(this.commLinesMesh);

    // --- Survivor markers ---
    // Tall thin cylinder as a beacon
    const beaconGeo = new THREE.CylinderGeometry(0.3, 1.5, 12, 8);
    this.survivorMaterial = new THREE.MeshStandardMaterial({
      color: SURVIVOR_COLOR,
      emissive: SURVIVOR_COLOR,
      emissiveIntensity: 0.5,
      roughness: 0.4,
      metalness: 0.6,
    });
    this.survivorMesh = new THREE.InstancedMesh(beaconGeo, this.survivorMaterial, maxSurvivors);
    this.survivorMesh.count = 0;
    this.survivorMesh.frustumCulled = false;
    this.scene.add(this.survivorMesh);

    // --- Evidence markers ---
    // Footprint = flat arrow along heading; debris = small box; signal_fire
    // = glowing cone. Each is instanced so we can ship dozens cheaply.
    this.buildEvidencePool("footprint", new THREE.ConeGeometry(0.8, 2.2, 4));
    this.buildEvidencePool("debris", new THREE.BoxGeometry(1.2, 1.2, 1.2));
    this.buildEvidencePool("signal_fire", new THREE.ConeGeometry(1.1, 3.0, 12));
  }

  private buildEvidencePool(kind: string, geometry: THREE.BufferGeometry): void {
    const color = new THREE.Color(EVIDENCE_COLORS[kind] ?? 0xffffff);
    const mat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.6,
      roughness: 0.4,
      metalness: 0.3,
    });
    const mesh = new THREE.InstancedMesh(geometry, mat, this.maxEvidence);
    mesh.count = 0;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.evidenceMeshes.set(kind, mesh);
    this.evidenceMaterials.set(kind, mat);
  }

  update(
    drones: DroneState[],
    commsLinks: [number, number][],
    survivors: SurvivorState[],
    dt: number,
    evidence?: EvidenceMarker[],
  ): void {
    this.updateCommLinks(drones, commsLinks);
    this.updateSurvivors(survivors, dt);
    this.updateEvidence(evidence ?? [], dt);
  }

  private updateEvidence(evidence: EvidenceMarker[], dt: number): void {
    this.evidencePulseTime += dt * EVIDENCE_PULSE_SPEED;
    const pulse = 0.5 + 0.5 * Math.sin(this.evidencePulseTime);

    // Pulse the signal-fire emissive brightly (active signal).
    const fireMat = this.evidenceMaterials.get("signal_fire");
    if (fireMat) fireMat.emissiveIntensity = 0.5 + pulse * 1.2;
    const footMat = this.evidenceMaterials.get("footprint");
    if (footMat) footMat.emissiveIntensity = 0.4 + pulse * 0.3;
    const debrisMat = this.evidenceMaterials.get("debris");
    if (debrisMat) debrisMat.emissiveIntensity = 0.35;

    // Bucket by kind, then lay them out.
    const buckets = new Map<string, EvidenceMarker[]>();
    for (const e of evidence) {
      const arr = buckets.get(e.kind) ?? [];
      arr.push(e);
      buckets.set(e.kind, arr);
    }

    for (const [kind, mesh] of this.evidenceMeshes.entries()) {
      const items = buckets.get(kind) ?? [];
      let count = 0;
      for (const ev of items) {
        if (count >= this.maxEvidence) break;
        // Lift the marker above terrain so it's not half-buried
        const liftY = ev.position[1] + 2 * EVIDENCE_SCALE;
        this.posVec.set(ev.position[0], liftY, ev.position[2]);

        // Footprint rotates to point along heading — use the protocol
        // convention (0 = +Z/north, clockwise). Cones default-point along
        // +Y; lay them on their side, then yaw to match heading.
        this.scaleVec.set(EVIDENCE_SCALE, EVIDENCE_SCALE, EVIDENCE_SCALE);
        if (kind === "footprint" && ev.heading !== null && ev.heading !== undefined) {
          // Tilt cone to face horizontal, then yaw.
          const euler = new THREE.Euler(Math.PI / 2, -ev.heading, 0, "YXZ");
          this.quat.setFromEuler(euler);
        } else {
          this.quat.identity();
        }
        this.mat4.compose(this.posVec, this.quat, this.scaleVec);
        mesh.setMatrixAt(count, this.mat4);
        count++;
      }
      mesh.count = count;
      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  private updateCommLinks(drones: DroneState[], links: [number, number][]): void {
    const droneMap = new Map<number, DroneState>();
    for (const d of drones) {
      droneMap.set(d.id, d);
    }

    let vertexIndex = 0;
    let linkCount = 0;
    for (const [idA, idB] of links) {
      if (linkCount >= this.maxLinks) break;
      const a = droneMap.get(idA);
      const b = droneMap.get(idB);
      if (!a || !b) continue;

      this.commPositions[vertexIndex++] = a.position[0];
      this.commPositions[vertexIndex++] = a.position[1];
      this.commPositions[vertexIndex++] = a.position[2];
      this.commPositions[vertexIndex++] = b.position[0];
      this.commPositions[vertexIndex++] = b.position[1];
      this.commPositions[vertexIndex++] = b.position[2];
      linkCount++;
    }

    const posAttr = this.commGeometry.getAttribute("position") as THREE.BufferAttribute;
    posAttr.needsUpdate = true;
    this.commGeometry.setDrawRange(0, linkCount * 2);
  }

  private updateSurvivors(survivors: SurvivorState[], dt: number): void {
    this.survivorPulseTime += dt * SURVIVOR_PULSE_SPEED;
    const pulse = 0.5 + 0.5 * Math.sin(this.survivorPulseTime);

    // Pulsing emissive
    this.survivorMaterial.emissiveIntensity = 0.3 + pulse * 0.7;

    let count = 0;
    for (const s of survivors) {
      if (count >= this.maxSurvivors) break;
      const scale = (0.8 + pulse * 0.4) * SURVIVOR_SCALE;
      this.mat4.makeScale(scale, scale, scale);
      this.mat4.setPosition(s.position[0], s.position[1] + 6 * SURVIVOR_SCALE, s.position[2]);
      this.survivorMesh.setMatrixAt(count, this.mat4);
      count++;
    }
    this.survivorMesh.count = count;
    this.survivorMesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    if (this.commLinesMesh) {
      this.scene.remove(this.commLinesMesh);
    }
    this.commGeometry.dispose();
    this.commMaterial.dispose();
    this.scene.remove(this.survivorMesh);
    this.survivorMesh.geometry.dispose();
    this.survivorMaterial.dispose();
    for (const mesh of this.evidenceMeshes.values()) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
    }
    for (const mat of this.evidenceMaterials.values()) {
      mat.dispose();
    }
    this.evidenceMeshes.clear();
    this.evidenceMaterials.clear();
  }
}
