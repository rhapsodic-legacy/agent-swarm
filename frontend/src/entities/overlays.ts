/**
 * Renders communication links between drones and survivor markers.
 */

import * as THREE from "three";
import type { DroneState, SurvivorState } from "@/network/types";

const COMM_LINK_COLOR = new THREE.Color(0x00aaff);
const COMM_LINK_OPACITY = 0.15;
const SURVIVOR_COLOR = new THREE.Color(0xff2200);
const SURVIVOR_PULSE_SPEED = 3.0;

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

  // Scratch objects
  private readonly mat4 = new THREE.Matrix4();

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
  }

  update(
    drones: DroneState[],
    commsLinks: [number, number][],
    survivors: SurvivorState[],
    dt: number,
  ): void {
    this.updateCommLinks(drones, commsLinks);
    this.updateSurvivors(survivors, dt);
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
      const scale = 0.8 + pulse * 0.4;
      this.mat4.makeScale(scale, scale, scale);
      this.mat4.setPosition(s.position[0], s.position[1] + 6, s.position[2]);
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
  }
}
