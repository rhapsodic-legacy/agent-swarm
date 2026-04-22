/**
 * ZoneTool — lets the operator draw priority-colored rectangles on the
 * terrain. Active mode is toggled with the `Z` key or the panel button.
 *
 * While active:
 *   - OrbitControls are disabled (so left-drag draws instead of orbiting).
 *   - Left-drag on the ground plane defines a rectangle (XZ axis-aligned).
 *   - On pointer up, a `zone_command` is sent to the backend.
 *   - Esc / right-click cancels the in-progress rectangle.
 *
 * The list of already-drawn zones is shown in the side panel with a
 * delete button per zone. The 3D rectangles themselves are rendered by
 * ZoneRenderer, driven by backend state (single source of truth).
 */

import * as THREE from "three";
import type { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { SwarmClient } from "@/network/client";
import type { ZoneData } from "@/network/types";

export type ZonePriority = "high" | "low" | "avoid";

const PRIORITY_COLORS: Record<ZonePriority, number> = {
  high: 0x44ff88,
  low: 0xffcc44,
  avoid: 0xff4466,
};

const PRIORITY_LABELS: Record<ZonePriority, string> = {
  high: "HIGH",
  low: "LOW",
  avoid: "AVOID",
};

const MIN_ZONE_SIZE_M = 50; // ignore tiny drag rectangles

function priorityHex(priority: ZonePriority): string {
  return "#" + PRIORITY_COLORS[priority].toString(16).padStart(6, "0");
}

export class ZoneTool {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private client: SwarmClient;

  private active = false;
  private priority: ZonePriority = "high";
  private zones: ZoneData[] = [];
  private zonesFingerprint = "";
  private nextZoneNumber = 1;

  // Preview rectangle (only shown while dragging)
  private previewMesh: THREE.Mesh;
  private previewMaterial: THREE.MeshBasicMaterial;
  private previewBorder: THREE.Line;
  private previewBorderMaterial: THREE.LineBasicMaterial;

  private raycaster = new THREE.Raycaster();
  private groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private dragStart: THREE.Vector3 | null = null;
  private dragEnd: THREE.Vector3 | null = null;

  private panel: HTMLDivElement;
  private indicator: HTMLDivElement;
  private zoneList: HTMLDivElement;

  private readonly handlePointerDown: (e: PointerEvent) => void;
  private readonly handlePointerMove: (e: PointerEvent) => void;
  private readonly handlePointerUp: (e: PointerEvent) => void;
  private readonly handleKeyDown: (e: KeyboardEvent) => void;
  private readonly handleContextMenu: (e: Event) => void;

  constructor(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    renderer: THREE.WebGLRenderer,
    controls: OrbitControls,
    client: SwarmClient,
  ) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.controls = controls;
    this.client = client;

    // Preview fill
    const fillGeo = new THREE.PlaneGeometry(1, 1);
    fillGeo.rotateX(-Math.PI / 2);
    this.previewMaterial = new THREE.MeshBasicMaterial({
      color: PRIORITY_COLORS[this.priority],
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false,
    });
    this.previewMesh = new THREE.Mesh(fillGeo, this.previewMaterial);
    this.previewMesh.position.y = 1.0;
    this.previewMesh.renderOrder = 1002;
    this.previewMesh.visible = false;
    this.scene.add(this.previewMesh);

    // Preview border
    const borderGeo = new THREE.BufferGeometry();
    borderGeo.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(5 * 3), 3),
    );
    this.previewBorderMaterial = new THREE.LineBasicMaterial({
      color: PRIORITY_COLORS[this.priority],
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    });
    this.previewBorder = new THREE.Line(borderGeo, this.previewBorderMaterial);
    this.previewBorder.renderOrder = 1003;
    this.previewBorder.visible = false;
    this.scene.add(this.previewBorder);

    this.panel = this.buildPanel();
    this.indicator = this.buildIndicator();
    this.zoneList = this.panel.querySelector("#zone-list") as HTMLDivElement;
    this.renderZoneList();

    this.handlePointerDown = this.onPointerDown.bind(this);
    this.handlePointerMove = this.onPointerMove.bind(this);
    this.handlePointerUp = this.onPointerUp.bind(this);
    this.handleKeyDown = this.onKeyDown.bind(this);
    this.handleContextMenu = (e: Event) => {
      if (this.active) e.preventDefault();
    };

    // Use capture phase so ZoneTool runs before InteractionManager's
    // bubble-phase handlers — letting us stopPropagation when zone mode
    // is active, so a left-drag paints a zone instead of moving a drone.
    const dom = this.renderer.domElement;
    dom.addEventListener("pointerdown", this.handlePointerDown, true);
    dom.addEventListener("pointermove", this.handlePointerMove, true);
    dom.addEventListener("pointerup", this.handlePointerUp, true);
    dom.addEventListener("contextmenu", this.handleContextMenu, true);
    window.addEventListener("keydown", this.handleKeyDown);
  }

  setZones(zones: ZoneData[] | undefined): void {
    const incoming = zones ?? [];
    const fp = incoming
      .map((z) => `${z.zone_id}:${z.priority}:${z.polygon.length}`)
      .join("|");
    if (fp === this.zonesFingerprint) return;
    this.zonesFingerprint = fp;
    this.zones = incoming;
    for (const z of this.zones) {
      const m = /^zone_(\d+)$/.exec(z.zone_id);
      if (m) this.nextZoneNumber = Math.max(this.nextZoneNumber, parseInt(m[1], 10) + 1);
    }
    this.renderZoneList();
  }

  toggle(): void {
    this.setActive(!this.active);
  }

  isActive(): boolean {
    return this.active;
  }

  private setActive(active: boolean): void {
    this.active = active;
    this.indicator.style.display = active ? "block" : "none";
    this.controls.enabled = !active;
    this.renderer.domElement.style.cursor = active ? "crosshair" : "";
    if (!active) this.cancelDrag();
    const btn = this.panel.querySelector("#zone-toggle") as HTMLButtonElement;
    if (btn) {
      btn.textContent = active ? "Drawing… (Z to stop)" : "Draw Zone (Z)";
      btn.style.background = active
        ? priorityHex(this.priority)
        : "rgba(0, 255, 255, 0.15)";
      btn.style.color = active ? "#000" : "#00ffff";
    }
  }

  private setPriority(priority: ZonePriority): void {
    this.priority = priority;
    const color = PRIORITY_COLORS[priority];
    this.previewMaterial.color.setHex(color);
    this.previewBorderMaterial.color.setHex(color);
    if (this.active) {
      const btn = this.panel.querySelector("#zone-toggle") as HTMLButtonElement;
      if (btn) btn.style.background = priorityHex(priority);
    }
    // Update radio UI
    for (const p of ["high", "low", "avoid"] as ZonePriority[]) {
      const el = this.panel.querySelector(`#prio-${p}`) as HTMLButtonElement | null;
      if (!el) continue;
      const selected = p === priority;
      el.style.background = selected
        ? priorityHex(p)
        : "rgba(40, 40, 60, 0.6)";
      el.style.color = selected ? "#000" : "#aaa";
      el.style.fontWeight = selected ? "700" : "500";
    }
  }

  private onKeyDown(event: KeyboardEvent): void {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "z") {
      event.preventDefault();
      this.toggle();
    } else if (event.key === "Escape" && this.active) {
      event.preventDefault();
      if (this.dragStart) {
        this.cancelDrag();
      } else {
        this.setActive(false);
      }
    }
  }

  private onPointerDown(event: PointerEvent): void {
    if (!this.active) return;
    if (event.button !== 0) {
      // Right-click cancels an in-progress drag.
      if (event.button === 2) this.cancelDrag();
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const hit = this.raycastGround(event);
    if (!hit) return;
    this.dragStart = hit.clone();
    this.dragEnd = hit.clone();
    this.updatePreview();
    this.previewMesh.visible = true;
    this.previewBorder.visible = true;
  }

  private onPointerMove(event: PointerEvent): void {
    if (!this.active || !this.dragStart) return;
    const hit = this.raycastGround(event);
    if (!hit) return;
    this.dragEnd = hit;
    this.updatePreview();
  }

  private onPointerUp(event: PointerEvent): void {
    if (!this.active || !this.dragStart) return;
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const end = this.raycastGround(event) ?? this.dragEnd;
    if (!end) {
      this.cancelDrag();
      return;
    }
    const width = Math.abs(end.x - this.dragStart.x);
    const depth = Math.abs(end.z - this.dragStart.z);
    if (width < MIN_ZONE_SIZE_M || depth < MIN_ZONE_SIZE_M) {
      this.cancelDrag();
      return;
    }
    const xMin = Math.min(this.dragStart.x, end.x);
    const xMax = Math.max(this.dragStart.x, end.x);
    const zMin = Math.min(this.dragStart.z, end.z);
    const zMax = Math.max(this.dragStart.z, end.z);
    const polygon: [number, number][] = [
      [xMin, zMin],
      [xMax, zMin],
      [xMax, zMax],
      [xMin, zMax],
    ];
    const zoneId = `zone_${this.nextZoneNumber++}`;
    this.client.sendZoneCommand({
      action: "create",
      zone_id: zoneId,
      polygon,
      priority: this.priority,
    });
    this.cancelDrag();
  }

  private cancelDrag(): void {
    this.dragStart = null;
    this.dragEnd = null;
    this.previewMesh.visible = false;
    this.previewBorder.visible = false;
  }

  private raycastGround(event: PointerEvent): THREE.Vector3 | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(this.groundPlane, hit)
      ? hit
      : null;
  }

  private updatePreview(): void {
    if (!this.dragStart || !this.dragEnd) return;
    const xMin = Math.min(this.dragStart.x, this.dragEnd.x);
    const xMax = Math.max(this.dragStart.x, this.dragEnd.x);
    const zMin = Math.min(this.dragStart.z, this.dragEnd.z);
    const zMax = Math.max(this.dragStart.z, this.dragEnd.z);
    const cx = (xMin + xMax) / 2;
    const cz = (zMin + zMax) / 2;
    const w = Math.max(xMax - xMin, 0.01);
    const d = Math.max(zMax - zMin, 0.01);
    this.previewMesh.position.set(cx, 1.0, cz);
    this.previewMesh.scale.set(w, 1, d);
    const positions = (
      this.previewBorder.geometry.attributes.position as THREE.BufferAttribute
    ).array as Float32Array;
    const y = 1.1;
    positions[0] = xMin; positions[1] = y; positions[2] = zMin;
    positions[3] = xMax; positions[4] = y; positions[5] = zMin;
    positions[6] = xMax; positions[7] = y; positions[8] = zMax;
    positions[9] = xMin; positions[10] = y; positions[11] = zMax;
    positions[12] = xMin; positions[13] = y; positions[14] = zMin;
    (
      this.previewBorder.geometry.attributes.position as THREE.BufferAttribute
    ).needsUpdate = true;
  }

  private buildPanel(): HTMLDivElement {
    const panel = document.createElement("div");
    panel.id = "zone-panel";
    Object.assign(panel.style, {
      position: "absolute",
      top: "80px",
      right: "16px",
      width: "220px",
      padding: "12px",
      background: "rgba(10, 10, 26, 0.88)",
      border: "1px solid rgba(0, 255, 255, 0.25)",
      borderRadius: "6px",
      color: "#e0e0e0",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      fontSize: "12px",
      lineHeight: "1.5",
      pointerEvents: "auto",
      zIndex: "90",
      backdropFilter: "blur(4px)",
      boxShadow: "0 2px 12px rgba(0, 0, 0, 0.4)",
    });

    panel.innerHTML = `
      <div style="font-size: 11px; color: #00ffff; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px; text-transform: uppercase;">
        Priority Zones
      </div>
      <div style="display: flex; gap: 4px; margin-bottom: 8px;">
        <button id="prio-high" style="${radioStyle()}">High</button>
        <button id="prio-low" style="${radioStyle()}">Low</button>
        <button id="prio-avoid" style="${radioStyle()}">Avoid</button>
      </div>
      <button id="zone-toggle" style="${toggleStyle()}">Draw Zone (Z)</button>
      <div id="zone-list" style="margin-top: 10px; max-height: 180px; overflow-y: auto;"></div>
      <button id="zone-clear" style="${clearStyle()}">Clear All</button>
      <div style="margin-top: 6px; font-size: 10px; color: #666;">
        Drag on terrain to paint. High draws drones in. Avoid blocks zone entirely.
      </div>
    `;

    document.body.appendChild(panel);

    (panel.querySelector("#prio-high") as HTMLButtonElement).addEventListener(
      "click", () => this.setPriority("high"),
    );
    (panel.querySelector("#prio-low") as HTMLButtonElement).addEventListener(
      "click", () => this.setPriority("low"),
    );
    (panel.querySelector("#prio-avoid") as HTMLButtonElement).addEventListener(
      "click", () => this.setPriority("avoid"),
    );
    (panel.querySelector("#zone-toggle") as HTMLButtonElement).addEventListener(
      "click", () => this.toggle(),
    );
    (panel.querySelector("#zone-clear") as HTMLButtonElement).addEventListener(
      "click", () => this.client.sendZoneCommand({ action: "clear" }),
    );

    // Mark "high" as the initial selection styling
    setTimeout(() => this.setPriority(this.priority), 0);

    return panel;
  }

  private buildIndicator(): HTMLDivElement {
    const el = document.createElement("div");
    Object.assign(el.style, {
      position: "absolute",
      top: "50px",
      left: "50%",
      transform: "translateX(-50%)",
      background: "rgba(0, 255, 255, 0.9)",
      color: "#000",
      padding: "6px 14px",
      borderRadius: "4px",
      fontSize: "11px",
      fontWeight: "700",
      letterSpacing: "1px",
      textTransform: "uppercase",
      display: "none",
      zIndex: "200",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      boxShadow: "0 2px 10px rgba(0, 0, 0, 0.4)",
    });
    el.textContent = "Zone Drawing — left-drag to paint, Esc to cancel";
    document.body.appendChild(el);
    return el;
  }

  private renderZoneList(): void {
    if (!this.zoneList) return;
    if (this.zones.length === 0) {
      this.zoneList.innerHTML =
        '<div style="color: #666; font-style: italic; font-size: 11px;">No zones yet.</div>';
      return;
    }
    this.zoneList.innerHTML = this.zones
      .map((z) => {
        const pts = z.polygon;
        const xs = pts.map((p) => p[0]);
        const zs = pts.map((p) => p[1]);
        const w = Math.max(...xs) - Math.min(...xs);
        const h = Math.max(...zs) - Math.min(...zs);
        const color = priorityHex(z.priority);
        return `
          <div style="display: flex; align-items: center; gap: 6px; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
            <span style="width: 10px; height: 10px; background: ${color}; border-radius: 2px; flex-shrink: 0;"></span>
            <span style="flex: 1; font-size: 11px;">${z.zone_id} · ${PRIORITY_LABELS[z.priority]} · ${w.toFixed(0)}×${h.toFixed(0)}m</span>
            <button class="zone-del" data-zone="${z.zone_id}" style="${delStyle()}">×</button>
          </div>
        `;
      })
      .join("");
    this.zoneList.querySelectorAll(".zone-del").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const target = e.currentTarget as HTMLButtonElement;
        const zid = target.getAttribute("data-zone");
        if (zid) {
          this.client.sendZoneCommand({ action: "delete", zone_id: zid });
        }
      });
    });
  }

  dispose(): void {
    const dom = this.renderer.domElement;
    dom.removeEventListener("pointerdown", this.handlePointerDown, true);
    dom.removeEventListener("pointermove", this.handlePointerMove, true);
    dom.removeEventListener("pointerup", this.handlePointerUp, true);
    dom.removeEventListener("contextmenu", this.handleContextMenu, true);
    window.removeEventListener("keydown", this.handleKeyDown);
    this.scene.remove(this.previewMesh);
    this.previewMesh.geometry.dispose();
    this.previewMaterial.dispose();
    this.scene.remove(this.previewBorder);
    this.previewBorder.geometry.dispose();
    this.previewBorderMaterial.dispose();
    this.panel.remove();
    this.indicator.remove();
    this.controls.enabled = true;
  }
}

function radioStyle(): string {
  return [
    "flex: 1",
    "padding: 6px 4px",
    "font-family: inherit",
    "font-size: 11px",
    "border: 1px solid rgba(255,255,255,0.15)",
    "border-radius: 3px",
    "cursor: pointer",
    "background: rgba(40, 40, 60, 0.6)",
    "color: #aaa",
    "text-transform: uppercase",
    "letter-spacing: 0.5px",
  ].join("; ");
}

function toggleStyle(): string {
  return [
    "width: 100%",
    "padding: 8px",
    "font-family: inherit",
    "font-size: 12px",
    "font-weight: 700",
    "border: 1px solid rgba(0, 255, 255, 0.5)",
    "border-radius: 4px",
    "cursor: pointer",
    "background: rgba(0, 255, 255, 0.15)",
    "color: #00ffff",
    "text-transform: uppercase",
    "letter-spacing: 0.5px",
  ].join("; ");
}

function clearStyle(): string {
  return [
    "width: 100%",
    "margin-top: 6px",
    "padding: 4px",
    "font-family: inherit",
    "font-size: 10px",
    "border: 1px solid rgba(255, 68, 68, 0.4)",
    "border-radius: 3px",
    "cursor: pointer",
    "background: rgba(255, 68, 68, 0.08)",
    "color: #ff8888",
    "text-transform: uppercase",
    "letter-spacing: 0.5px",
  ].join("; ");
}

function delStyle(): string {
  return [
    "width: 20px",
    "height: 20px",
    "padding: 0",
    "border: 1px solid rgba(255,255,255,0.2)",
    "border-radius: 3px",
    "cursor: pointer",
    "background: rgba(0,0,0,0.3)",
    "color: #ff8888",
    "font-size: 14px",
    "line-height: 1",
    "flex-shrink: 0",
  ].join("; ");
}
