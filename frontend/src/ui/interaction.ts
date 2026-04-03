/**
 * InteractionManager — handles all human interaction with the 3D drone swarm scene.
 *
 * Responsibilities:
 *  - Drone selection via left-click (screen-space proximity picking)
 *  - Click-to-move via right-click or shift+click (raycast against ground plane)
 *  - Selection HUD overlay (drone info panel)
 *  - Terrain click indicators (pulsing target markers)
 */

import * as THREE from "three";
import type { DroneState } from "@/network/types";
import type { SwarmClient } from "@/network/client";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum screen-space distance (pixels) for drone picking. */
const PICK_RADIUS_PX = 30;

/** Duration (seconds) for the move-target indicator before it fades and is removed. */
const INDICATOR_LIFETIME = 2.0;

/** Selection ring visual properties. */
const RING_INNER_RADIUS = 2.0;
const RING_OUTER_RADIUS = 3.0;
const RING_SEGMENTS = 32;
const RING_COLOR = 0x00ffff;
const RING_OPACITY = 0.7;
const RING_PULSE_SPEED = 3.0;
const RING_PULSE_AMPLITUDE = 0.15;
const RING_HOVER_OFFSET = 0.3;

/** Move indicator visual properties. */
const INDICATOR_RADIUS = 1.5;
const INDICATOR_COLOR = 0x00ffff;
const INDICATOR_PULSE_SPEED = 6.0;
const INDICATOR_PULSE_AMPLITUDE = 0.4;

/** HUD styling. */
const HUD_ELEMENT_ID = "selection-info";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface MoveIndicator {
  mesh: THREE.Mesh;
  elapsed: number;
  lifetime: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createSelectionRing(): THREE.Mesh {
  const geometry = new THREE.RingGeometry(
    RING_INNER_RADIUS,
    RING_OUTER_RADIUS,
    RING_SEGMENTS,
  );
  // Rotate ring to lie flat on XZ plane (default RingGeometry faces +Z).
  geometry.rotateX(-Math.PI / 2);

  const material = new THREE.MeshBasicMaterial({
    color: RING_COLOR,
    transparent: true,
    opacity: RING_OPACITY,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 10;
  mesh.visible = false;
  return mesh;
}

function createMoveIndicatorMesh(): THREE.Mesh {
  const geometry = new THREE.CircleGeometry(INDICATOR_RADIUS, 24);
  // Lie flat on XZ plane.
  geometry.rotateX(-Math.PI / 2);

  const material = new THREE.MeshBasicMaterial({
    color: INDICATOR_COLOR,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 10;
  return mesh;
}

function ensureHUDElement(): HTMLDivElement {
  let el = document.getElementById(HUD_ELEMENT_ID) as HTMLDivElement | null;
  if (el) return el;

  el = document.createElement("div");
  el.id = HUD_ELEMENT_ID;

  Object.assign(el.style, {
    position: "absolute",
    bottom: "16px",
    left: "16px",
    padding: "12px 16px",
    background: "rgba(10, 10, 26, 0.85)",
    border: "1px solid rgba(0, 255, 255, 0.3)",
    borderRadius: "6px",
    color: "#e0e0e0",
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    fontSize: "13px",
    lineHeight: "1.5",
    minWidth: "200px",
    pointerEvents: "none",
    zIndex: "100",
    display: "none",
    backdropFilter: "blur(4px)",
    boxShadow: "0 2px 12px rgba(0, 0, 0, 0.4)",
  } satisfies Partial<CSSStyleDeclaration>);

  document.body.appendChild(el);
  return el;
}

/**
 * Project a 3D world position to 2D screen coordinates.
 * Returns pixel coordinates where (0,0) is top-left of the canvas.
 */
function projectToScreen(
  worldPos: THREE.Vector3,
  camera: THREE.PerspectiveCamera,
  canvasWidth: number,
  canvasHeight: number,
): { x: number; y: number } {
  const projected = worldPos.clone().project(camera);
  return {
    x: (projected.x * 0.5 + 0.5) * canvasWidth,
    y: (-projected.y * 0.5 + 0.5) * canvasHeight,
  };
}

/**
 * Convert a mouse/pointer event to normalized device coordinates [-1, 1].
 */
function eventToNDC(
  event: MouseEvent,
  domElement: HTMLElement,
): THREE.Vector2 {
  const rect = domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
}

// ---------------------------------------------------------------------------
// InteractionManager
// ---------------------------------------------------------------------------

export class InteractionManager {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private client: SwarmClient;
  private domElement: HTMLElement;

  // --- State ---
  private drones: DroneState[] = [];
  private selectedDroneId: number | null = null;

  // --- Three.js objects ---
  private raycaster = new THREE.Raycaster();
  private groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private selectionRing: THREE.Mesh;
  private moveIndicators: MoveIndicator[] = [];

  // --- HUD ---
  private hudElement: HTMLDivElement;

  // --- Timing ---
  private clock = new THREE.Clock();

  // --- Bound event handlers (stored for removal in dispose) ---
  private readonly handlePointerDown: (e: PointerEvent) => void;
  private readonly handleContextMenu: (e: Event) => void;

  constructor(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    renderer: THREE.WebGLRenderer,
    client: SwarmClient,
  ) {
    this.scene = scene;
    this.camera = camera;
    this.client = client;
    this.domElement = renderer.domElement;

    // Create visual elements.
    this.selectionRing = createSelectionRing();
    this.scene.add(this.selectionRing);

    this.hudElement = ensureHUDElement();

    // Bind event handlers.
    this.handlePointerDown = this.onPointerDown.bind(this);
    this.handleContextMenu = (e: Event) => e.preventDefault();

    this.domElement.addEventListener("pointerdown", this.handlePointerDown);
    this.domElement.addEventListener("contextmenu", this.handleContextMenu);
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Call every frame with the latest drone state array.
   * Updates the selection ring position, move indicators, and HUD.
   */
  update(drones: DroneState[]): void {
    this.drones = drones;
    const dt = this.clock.getDelta();

    this.updateSelectionRing();
    this.updateMoveIndicators(dt);
    this.updateHUD();
  }

  /** Get the currently selected drone ID, or null if nothing is selected. */
  getSelectedDroneId(): number | null {
    return this.selectedDroneId;
  }

  /** Clean up all event listeners, Three.js objects, and DOM elements. */
  dispose(): void {
    this.domElement.removeEventListener("pointerdown", this.handlePointerDown);
    this.domElement.removeEventListener("contextmenu", this.handleContextMenu);

    // Selection ring.
    this.scene.remove(this.selectionRing);
    (this.selectionRing.geometry as THREE.BufferGeometry).dispose();
    (this.selectionRing.material as THREE.Material).dispose();

    // Move indicators.
    for (const indicator of this.moveIndicators) {
      this.scene.remove(indicator.mesh);
      (indicator.mesh.geometry as THREE.BufferGeometry).dispose();
      (indicator.mesh.material as THREE.Material).dispose();
    }
    this.moveIndicators.length = 0;

    // HUD element.
    if (this.hudElement.parentNode) {
      this.hudElement.parentNode.removeChild(this.hudElement);
    }

    this.selectedDroneId = null;
    this.drones = [];
  }

  // -----------------------------------------------------------------------
  // Event Handlers
  // -----------------------------------------------------------------------

  private onPointerDown(event: PointerEvent): void {
    // Ignore middle mouse button and other non-standard buttons.
    const isLeftClick = event.button === 0 && !event.shiftKey;
    const isShiftClick = event.button === 0 && event.shiftKey;
    const isRightClick = event.button === 2;

    if (isLeftClick) {
      this.handleSelect(event);
    } else if (isRightClick || isShiftClick) {
      this.handleMoveCommand(event);
    }
  }

  // -----------------------------------------------------------------------
  // Selection (Left Click)
  // -----------------------------------------------------------------------

  private handleSelect(event: PointerEvent): void {
    const canvasWidth = this.domElement.clientWidth;
    const canvasHeight = this.domElement.clientHeight;
    const rect = this.domElement.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    let closestId: number | null = null;
    let closestDist = PICK_RADIUS_PX;

    for (const drone of this.drones) {
      const worldPos = new THREE.Vector3(
        drone.position[0],
        drone.position[1],
        drone.position[2],
      );

      // Cull drones behind the camera.
      const toCamera = worldPos.clone().sub(this.camera.position);
      const cameraDir = new THREE.Vector3();
      this.camera.getWorldDirection(cameraDir);
      if (toCamera.dot(cameraDir) < 0) continue;

      const screen = projectToScreen(worldPos, this.camera, canvasWidth, canvasHeight);
      const dx = screen.x - mouseX;
      const dy = screen.y - mouseY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < closestDist) {
        closestDist = dist;
        closestId = drone.id;
      }
    }

    this.selectedDroneId = closestId;

    if (closestId === null) {
      this.selectionRing.visible = false;
      this.hudElement.style.display = "none";
    }
  }

  // -----------------------------------------------------------------------
  // Click-to-Move (Right Click / Shift+Click)
  // -----------------------------------------------------------------------

  private handleMoveCommand(event: PointerEvent): void {
    if (this.selectedDroneId === null) return;

    const ndc = eventToNDC(event, this.domElement);
    this.raycaster.setFromCamera(ndc, this.camera);

    // Raycast against the Y=0 ground plane.
    const intersection = new THREE.Vector3();
    const hit = this.raycaster.ray.intersectPlane(this.groundPlane, intersection);
    if (!hit) return;

    // Send command to backend. The protocol expects [x, z].
    this.client.sendCommand("move_to", this.selectedDroneId, [
      intersection.x,
      intersection.z,
    ]);

    // Spawn a visual indicator at the target.
    this.spawnMoveIndicator(intersection);
  }

  private spawnMoveIndicator(position: THREE.Vector3): void {
    const mesh = createMoveIndicatorMesh();
    // Place slightly above ground to avoid z-fighting.
    mesh.position.set(position.x, 0.05, position.z);
    this.scene.add(mesh);

    this.moveIndicators.push({
      mesh,
      elapsed: 0,
      lifetime: INDICATOR_LIFETIME,
    });
  }

  // -----------------------------------------------------------------------
  // Per-Frame Updates
  // -----------------------------------------------------------------------

  private updateSelectionRing(): void {
    if (this.selectedDroneId === null) {
      this.selectionRing.visible = false;
      return;
    }

    // Find the selected drone's current state.
    const drone = this.drones.find((d) => d.id === this.selectedDroneId);
    if (!drone) {
      // Drone disappeared from state — deselect.
      this.selectedDroneId = null;
      this.selectionRing.visible = false;
      this.hudElement.style.display = "none";
      return;
    }

    this.selectionRing.visible = true;

    // Position the ring below the drone (on the ground, or just beneath it).
    this.selectionRing.position.set(
      drone.position[0],
      RING_HOVER_OFFSET,
      drone.position[2],
    );

    // Pulse scale.
    const time = performance.now() / 1000;
    const pulse = 1.0 + Math.sin(time * RING_PULSE_SPEED) * RING_PULSE_AMPLITUDE;
    this.selectionRing.scale.set(pulse, 1, pulse);

    // Pulse opacity.
    const material = this.selectionRing.material as THREE.MeshBasicMaterial;
    material.opacity =
      RING_OPACITY - Math.sin(time * RING_PULSE_SPEED) * 0.15;
  }

  private updateMoveIndicators(dt: number): void {
    const toRemove: number[] = [];

    for (let i = 0; i < this.moveIndicators.length; i++) {
      const indicator = this.moveIndicators[i];
      indicator.elapsed += dt;

      const progress = indicator.elapsed / indicator.lifetime;

      if (progress >= 1.0) {
        toRemove.push(i);
        continue;
      }

      // Pulse scale.
      const time = performance.now() / 1000;
      const pulse =
        1.0 + Math.sin(time * INDICATOR_PULSE_SPEED) * INDICATOR_PULSE_AMPLITUDE;
      indicator.mesh.scale.set(pulse, 1, pulse);

      // Fade out opacity over the lifetime.
      const material = indicator.mesh.material as THREE.MeshBasicMaterial;
      material.opacity = 0.8 * (1.0 - progress);
    }

    // Remove expired indicators in reverse order to preserve indices.
    for (let i = toRemove.length - 1; i >= 0; i--) {
      const idx = toRemove[i];
      const indicator = this.moveIndicators[idx];
      this.scene.remove(indicator.mesh);
      (indicator.mesh.geometry as THREE.BufferGeometry).dispose();
      (indicator.mesh.material as THREE.Material).dispose();
      this.moveIndicators.splice(idx, 1);
    }
  }

  private updateHUD(): void {
    if (this.selectedDroneId === null) {
      this.hudElement.style.display = "none";
      return;
    }

    const drone = this.drones.find((d) => d.id === this.selectedDroneId);
    if (!drone) {
      this.hudElement.style.display = "none";
      return;
    }

    this.hudElement.style.display = "block";

    const statusColor = getStatusColor(drone.status);
    const batteryColor = getBatteryColor(drone.battery);
    const pos = drone.position;

    this.hudElement.innerHTML = `
      <div style="margin-bottom: 6px; font-size: 14px; color: #00ffff; font-weight: bold;">
        Drone #${drone.id}
      </div>
      <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 10px;">
        <span style="color: #888;">Status</span>
        <span style="color: ${statusColor};">${drone.status.toUpperCase()}</span>
        <span style="color: #888;">Battery</span>
        <span style="color: ${batteryColor};">${drone.battery.toFixed(0)}%</span>
        <span style="color: #888;">Task</span>
        <span>${drone.current_task || "idle"}</span>
        <span style="color: #888;">Position</span>
        <span>${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)}</span>
        ${
          drone.target
            ? `<span style="color: #888;">Target</span>
               <span>${drone.target[0].toFixed(1)}, ${drone.target[1].toFixed(1)}, ${drone.target[2].toFixed(1)}</span>`
            : ""
        }
        <span style="color: #888;">Sensor</span>
        <span>${drone.sensor_active ? "ON" : "OFF"}</span>
        <span style="color: #888;">Comms</span>
        <span>${drone.comms_active ? "ON" : "OFF"}</span>
      </div>
    `;
  }
}

// ---------------------------------------------------------------------------
// HUD color helpers
// ---------------------------------------------------------------------------

function getStatusColor(status: DroneState["status"]): string {
  switch (status) {
    case "active":
      return "#44ff44";
    case "returning":
      return "#ffcc00";
    case "recharging":
      return "#00ccff";
    case "failed":
      return "#ff4444";
  }
}

function getBatteryColor(battery: number): string {
  if (battery > 60) return "#44ff44";
  if (battery > 30) return "#ffcc00";
  return "#ff4444";
}
