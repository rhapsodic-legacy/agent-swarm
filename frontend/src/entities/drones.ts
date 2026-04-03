/**
 * DroneRenderer — InstancedMesh-based renderer for the drone swarm.
 *
 * Renders all drones in a single draw call via InstancedMesh, with per-instance
 * coloring by status, smooth position interpolation, trailing paths, and optional
 * sensor cones for active drones.
 */

import * as THREE from "three";
import type { DroneState } from "@/network/types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SENSOR_RANGE = 40;
const TRAIL_LENGTH = 30;
const LERP_RATE = 10; // positions converge at ~10/sec

const STATUS_COLORS: Record<DroneState["status"], THREE.Color> = {
  active: new THREE.Color(0x44ff44),
  returning: new THREE.Color(0xffcc00),
  recharging: new THREE.Color(0x00ccff),
  failed: new THREE.Color(0xff4444),
};

const TRAIL_COLOR = new THREE.Color(0xffffff);

// ---------------------------------------------------------------------------
// Internal per-drone bookkeeping
// ---------------------------------------------------------------------------

interface DroneSlot {
  /** Smoothed (rendered) position — lerped toward the authoritative target. */
  rendered: THREE.Vector3;
  /** Most recent authoritative position from the simulation. */
  target: THREE.Vector3;
  /** Ring-buffer of recent positions for the trail. */
  trail: THREE.Vector3[];
  /** Index into the ring-buffer for the next write. */
  trailHead: number;
  /** Whether we have received at least one position for this drone. */
  initialised: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const _mat4 = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _axisY = new THREE.Vector3(0, 1, 0);
const _axisX = new THREE.Vector3(1, 0, 0);
const _color = new THREE.Color();

function makeDroneGeometry(): THREE.OctahedronGeometry {
  return new THREE.OctahedronGeometry(1.5, 0);
}

function makeDroneMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    metalness: 0.5,
    roughness: 0.3,
  });
}

function makeSensorConeGeometry(): THREE.ConeGeometry {
  // Cone points downward. We create it pointing along +Y, then rotate in the
  // instance matrix so the tip is at drone altitude and the open base is on the
  // ground. Height = droneAltitude is unknown at construction time, so we use a
  // unit-height cone and scale per-instance.
  return new THREE.ConeGeometry(SENSOR_RANGE, 1, 16, 1, true);
}

function makeSensorConeMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: 0x00ff88,
    transparent: true,
    opacity: 0.05,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
}

function makeTrailMaterial(): THREE.LineBasicMaterial {
  return new THREE.LineBasicMaterial({
    color: TRAIL_COLOR,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
  });
}

// ---------------------------------------------------------------------------
// DroneRenderer
// ---------------------------------------------------------------------------

export class DroneRenderer {
  private scene: THREE.Scene;
  private maxDrones: number;

  // --- Meshes & materials (created once) ---
  private droneGeometry: THREE.OctahedronGeometry;
  private droneMaterial: THREE.MeshStandardMaterial;
  private droneMesh: THREE.InstancedMesh;

  private coneGeometry: THREE.ConeGeometry;
  private coneMaterial: THREE.MeshStandardMaterial;
  private coneMesh: THREE.InstancedMesh;

  // --- Trails ---
  private trailGroup: THREE.Group;
  private trailMaterial: THREE.LineBasicMaterial;
  /** Indexed by slot (same index as InstancedMesh). */
  private trailLines: (THREE.Line | null)[];
  private trailGeometries: (THREE.BufferGeometry | null)[];

  // --- Per-drone state ---
  /** Map from drone ID to slot index. */
  private idToSlot: Map<number, number> = new Map();
  private slots: (DroneSlot | null)[];

  // -----------------------------------------------------------------------
  constructor(scene: THREE.Scene, maxDrones: number) {
    this.scene = scene;
    this.maxDrones = maxDrones;

    // --- Drone instanced mesh ---
    this.droneGeometry = makeDroneGeometry();
    this.droneMaterial = makeDroneMaterial();
    this.droneMesh = new THREE.InstancedMesh(this.droneGeometry, this.droneMaterial, maxDrones);
    this.droneMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.droneMesh.count = 0;
    this.droneMesh.frustumCulled = false;
    scene.add(this.droneMesh);

    // --- Sensor cone instanced mesh ---
    this.coneGeometry = makeSensorConeGeometry();
    this.coneMaterial = makeSensorConeMaterial();
    this.coneMesh = new THREE.InstancedMesh(this.coneGeometry, this.coneMaterial, maxDrones);
    this.coneMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.coneMesh.count = 0;
    this.coneMesh.frustumCulled = false;
    this.coneMesh.renderOrder = -1; // render before opaque to reduce artifacts
    scene.add(this.coneMesh);

    // --- Trails ---
    this.trailGroup = new THREE.Group();
    scene.add(this.trailGroup);
    this.trailMaterial = makeTrailMaterial();
    this.trailLines = new Array<THREE.Line | null>(maxDrones).fill(null);
    this.trailGeometries = new Array<THREE.BufferGeometry | null>(maxDrones).fill(null);

    // --- Slots ---
    this.slots = new Array<DroneSlot | null>(maxDrones).fill(null);
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Update all drone instances from the latest simulation snapshot.
   * @param drones Array of current drone states from the backend.
   * @param dt     Seconds since the last frame.
   */
  update(drones: DroneState[], dt: number): void {
    const lerpFactor = 1 - Math.exp(-LERP_RATE * dt);

    // Track which slots are still alive this frame so we can hide stale ones.
    const aliveSlots = new Set<number>();

    let coneIndex = 0;

    for (const drone of drones) {
      const slot = this.ensureSlot(drone.id);
      const idx = this.idToSlot.get(drone.id)!;
      aliveSlots.add(idx);

      // Update authoritative target position.
      slot.target.set(drone.position[0], drone.position[1], drone.position[2]);

      if (!slot.initialised) {
        // First time — snap directly, no lerp.
        slot.rendered.copy(slot.target);
        slot.initialised = true;
      } else {
        slot.rendered.lerp(slot.target, lerpFactor);
      }

      // --- Instance matrix ---
      const scale = drone.status === "failed" ? 0.5 : 1.0;
      _quat.setFromAxisAngle(_axisY, drone.heading);
      _scale.set(scale, scale, scale);
      _mat4.compose(slot.rendered, _quat, _scale);
      this.droneMesh.setMatrixAt(idx, _mat4);

      // --- Instance color ---
      _color.copy(STATUS_COLORS[drone.status]);
      this.droneMesh.setColorAt(idx, _color);

      // --- Sensor cone ---
      if (drone.status === "active" && drone.sensor_active) {
        const altitude = Math.max(slot.rendered.y, 1); // avoid zero-height
        // Cone: tip at drone position, opening downward. ConeGeometry is centred
        // on its origin along +Y; we want it hanging below the drone. The cone
        // has unit height; we scale Y to the altitude so the base reaches Y=0.
        // We also rotate it PI around X so the open end faces down.
        _pos.copy(slot.rendered);
        _pos.y -= altitude / 2; // centre of cone is halfway down
        _quat.setFromAxisAngle(_axisX, Math.PI);
        _scale.set(1, altitude, 1);
        _mat4.compose(_pos, _quat, _scale);
        this.coneMesh.setMatrixAt(coneIndex, _mat4);
        coneIndex++;
      }

      // --- Trail ---
      if (drone.status !== "failed") {
        this.updateTrail(idx, slot);
      }
    }

    // --- Update instance counts ---
    this.droneMesh.count = drones.length;
    this.droneMesh.instanceMatrix.needsUpdate = true;
    if (this.droneMesh.instanceColor) {
      this.droneMesh.instanceColor.needsUpdate = true;
    }

    this.coneMesh.count = coneIndex;
    this.coneMesh.instanceMatrix.needsUpdate = true;

    // --- Hide trails for drones no longer present ---
    for (let i = 0; i < this.maxDrones; i++) {
      if (!aliveSlots.has(i) && this.trailLines[i]) {
        this.trailLines[i]!.visible = false;
      }
    }
  }

  /**
   * Retrieve the smoothed world-space position of a drone by its ID.
   * Returns null if the drone is not currently tracked.
   */
  getDronePosition(id: number): THREE.Vector3 | null {
    const idx = this.idToSlot.get(id);
    if (idx === undefined) return null;
    const slot = this.slots[idx];
    if (!slot) return null;
    return slot.rendered.clone();
  }

  /** Remove all Three.js objects and release GPU resources. */
  dispose(): void {
    // Drone mesh
    this.scene.remove(this.droneMesh);
    this.droneGeometry.dispose();
    this.droneMaterial.dispose();

    // Sensor cones
    this.scene.remove(this.coneMesh);
    this.coneGeometry.dispose();
    this.coneMaterial.dispose();

    // Trails
    this.trailMaterial.dispose();
    for (let i = 0; i < this.maxDrones; i++) {
      if (this.trailGeometries[i]) {
        this.trailGeometries[i]!.dispose();
      }
      if (this.trailLines[i]) {
        this.trailGroup.remove(this.trailLines[i]!);
      }
    }
    this.scene.remove(this.trailGroup);

    this.idToSlot.clear();
    this.slots.fill(null);
    this.trailLines.fill(null);
    this.trailGeometries.fill(null);
  }

  // -----------------------------------------------------------------------
  // Internals
  // -----------------------------------------------------------------------

  /**
   * Ensure a slot exists for the given drone ID and return it.
   * Allocates a new slot if this is the first time we've seen this ID.
   */
  private ensureSlot(id: number): DroneSlot {
    let idx = this.idToSlot.get(id);
    if (idx !== undefined && this.slots[idx]) {
      return this.slots[idx]!;
    }

    // Allocate the next free slot.
    idx = this.idToSlot.size;
    if (idx >= this.maxDrones) {
      // Overflow — reuse slot 0 as a fallback (should not happen in practice).
      idx = 0;
    }
    this.idToSlot.set(id, idx);

    const slot: DroneSlot = {
      rendered: new THREE.Vector3(),
      target: new THREE.Vector3(),
      trail: [],
      trailHead: 0,
      initialised: false,
    };
    this.slots[idx] = slot;
    return slot;
  }

  /**
   * Push the current rendered position into the trail ring-buffer and update
   * the corresponding Line geometry.
   */
  private updateTrail(idx: number, slot: DroneSlot): void {
    // Push position into ring buffer.
    if (slot.trail.length < TRAIL_LENGTH) {
      slot.trail.push(slot.rendered.clone());
    } else {
      slot.trail[slot.trailHead].copy(slot.rendered);
    }
    slot.trailHead = (slot.trailHead + 1) % TRAIL_LENGTH;

    // Build ordered array from ring buffer (oldest → newest).
    const count = slot.trail.length;
    const ordered: THREE.Vector3[] = [];
    for (let i = 0; i < count; i++) {
      const ri = (slot.trailHead + i) % count;
      ordered.push(slot.trail[ri]);
    }

    // Create or update BufferGeometry.
    if (!this.trailGeometries[idx]) {
      const positions = new Float32Array(TRAIL_LENGTH * 3);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geom.setDrawRange(0, 0);
      this.trailGeometries[idx] = geom;

      const line = new THREE.Line(geom, this.trailMaterial);
      line.frustumCulled = false;
      this.trailLines[idx] = line;
      this.trailGroup.add(line);
    }

    const geom = this.trailGeometries[idx]!;
    const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;

    for (let i = 0; i < ordered.length; i++) {
      const v = ordered[i];
      arr[i * 3] = v.x;
      arr[i * 3 + 1] = v.y;
      arr[i * 3 + 2] = v.z;
    }
    posAttr.needsUpdate = true;
    geom.setDrawRange(0, ordered.length);

    this.trailLines[idx]!.visible = true;
  }
}
