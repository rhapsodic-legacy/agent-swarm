/**
 * God Mode overlay — reveals all survivor positions through fog.
 * Toggle with G key. Shows undiscovered survivors as dim markers,
 * discovered as bright markers.
 */

import * as THREE from "three";
import type { SurvivorState } from "@/network/types";

const UNDISCOVERED_COLOR = new THREE.Color(0xff2200);
const DISCOVERED_COLOR = new THREE.Color(0x00ff44);
const MARKER_SCALE = 5; // ~3-4x drone size (drones are ~24m with DRONE_SCALE=8)
const PULSE_SPEED = 2.5;

export class GodModeOverlay {
  private scene: THREE.Scene;
  private mesh: THREE.InstancedMesh;
  private material: THREE.MeshBasicMaterial;
  private active = false;
  private readonly maxMarkers = 500;
  private readonly mat4 = new THREE.Matrix4();
  private readonly color = new THREE.Color();
  private pulseTime = 0;

  // HUD indicator
  private indicator: HTMLDivElement;

  constructor(scene: THREE.Scene) {
    this.scene = scene;

    // Tall cylinder — visible as a vertical beam from far away
    const geo = new THREE.CylinderGeometry(0.5, 0.5, 5, 6);
    geo.translate(0, 2.5, 0); // base at ground
    // MeshBasicMaterial ignores lighting and tone mapping — always bright
    this.material = new THREE.MeshBasicMaterial({
      color: UNDISCOVERED_COLOR,
      transparent: true,
      opacity: 0.9,
      depthTest: false, // always render on top
    });

    this.mesh = new THREE.InstancedMesh(geo, this.material, this.maxMarkers);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    this.scene.add(this.mesh);

    // On-screen indicator
    this.indicator = document.createElement("div");
    Object.assign(this.indicator.style, {
      position: "absolute",
      top: "50px",
      right: "16px",
      background: "rgba(255, 68, 0, 0.8)",
      color: "#fff",
      padding: "4px 10px",
      borderRadius: "4px",
      fontSize: "11px",
      fontWeight: "700",
      letterSpacing: "1px",
      textTransform: "uppercase",
      display: "none",
      zIndex: "200",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
    });
    this.indicator.textContent = "God Mode";
    document.getElementById("app")!.appendChild(this.indicator);
  }

  toggle(): void {
    this.active = !this.active;
    this.mesh.visible = this.active;
    this.indicator.style.display = this.active ? "block" : "none";
    console.log(`[GodMode] ${this.active ? "ON" : "OFF"}, mesh count=${this.mesh.count}`);
  }

  isActive(): boolean {
    return this.active;
  }

  // Simple standalone meshes as fallback (InstancedMesh can have issues)
  private standaloneMarkers: THREE.Mesh[] = [];
  private markersBuilt = false;
  private _lastLogTick = 0;

  update(allSurvivors: SurvivorState[] | undefined, dt: number): void {
    if (!this.active || !allSurvivors || allSurvivors.length === 0) {
      if (this.active && this._lastLogTick++ % 100 === 0) {
        console.log(`[GodMode] active but no survivors data. allSurvivors=${allSurvivors?.length ?? 'undefined'}`);
      }
      this.mesh.count = 0;
      // Hide standalone markers
      for (const m of this.standaloneMarkers) m.visible = false;
      return;
    }

    if (this._lastLogTick++ % 200 === 0) {
      console.log(`[GodMode] rendering ${allSurvivors.length} survivors, first at (${allSurvivors[0]?.position?.join(',')})`);
    }

    // Rebuild markers when survivor count changes (new chunks loaded = new survivors visible)
    if (!this.markersBuilt || this.standaloneMarkers.length !== allSurvivors.length) {
      // Clear old markers
      for (const m of this.standaloneMarkers) {
        this.scene.remove(m);
        (m.material as THREE.Material).dispose();
      }
      this.standaloneMarkers = [];
      this.markersBuilt = true;
      // Markers should be ~3-4x drone visual size.
      // Drone geometry: OctahedronGeometry(1.5) — no additional scale applied here,
      // DRONE_SCALE is applied per-instance via matrix. Base geometry radius = 1.5.
      // Marker: 6m tall, 1.5m wide cylinder. Scaled by the same per-instance logic.
      const markerGeo = new THREE.CylinderGeometry(3, 3, 12, 6);
      for (const s of allSurvivors) {
        if (this.standaloneMarkers.length >= this.maxMarkers) break;
        const mat = new THREE.MeshBasicMaterial({
          color: s.discovered ? DISCOVERED_COLOR : UNDISCOVERED_COLOR,
          transparent: true,
          opacity: 0.85,
        });
        const marker = new THREE.Mesh(markerGeo, mat);
        marker.position.set(s.position[0], s.position[1] + 3, s.position[2]);
        marker.renderOrder = 999;
        this.scene.add(marker);
        this.standaloneMarkers.push(marker);
      }
      console.log(`[GodMode] Built ${this.standaloneMarkers.length} standalone markers`);
    }

    // Show standalone markers
    this.pulseTime += dt * PULSE_SPEED;
    const pulse = 0.7 + 0.3 * Math.sin(this.pulseTime);
    for (let i = 0; i < this.standaloneMarkers.length; i++) {
      const m = this.standaloneMarkers[i];
      m.visible = true;
      m.scale.set(pulse, 1, pulse);
    }

    // Also update instanced mesh (original path)
    let count = 0;
    for (const s of allSurvivors) {
      if (count >= this.maxMarkers) break;
      const scale = MARKER_SCALE * (s.discovered ? 1.2 : pulse);
      this.mat4.makeScale(scale, scale, scale);
      this.mat4.setPosition(
        s.position[0],
        s.position[1] + MARKER_SCALE * 3,
        s.position[2],
      );
      this.mesh.setMatrixAt(count, this.mat4);
      this.color.copy(s.discovered ? DISCOVERED_COLOR : UNDISCOVERED_COLOR);
      this.mesh.setColorAt(count, this.color);
      count++;
    }
    this.mesh.count = count;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) {
      this.mesh.instanceColor.needsUpdate = true;
    }
  }

  dispose(): void {
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.material.dispose();
    for (const m of this.standaloneMarkers) {
      this.scene.remove(m);
      (m.material as THREE.Material).dispose();
    }
    this.standaloneMarkers = [];
    this.markersBuilt = false;
    this.indicator.remove();
  }
}
