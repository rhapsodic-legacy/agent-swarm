import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SwarmClient } from "@/network/client";
import { decodeTerrain } from "@/network/types";
import type { HazardInfo, StateUpdate, ChunkTerrainData, WorldOverview } from "@/network/types";
import { TerrainRenderer } from "@/scene/terrain";
import { DroneRenderer } from "@/entities/drones";
import { FogOfWarRenderer } from "@/fog/fogOfWar";
import { OverlayRenderer } from "@/entities/overlays";
import { InteractionManager } from "@/ui/interaction";
import { ChatPanel } from "@/ui/chatPanel";
import { SettingsPanel } from "@/ui/settingsPanel";
import type { SimSettings } from "@/ui/settingsPanel";
import { Minimap } from "@/ui/minimap";
import { ActivityLog } from "@/ui/activityLog";
import { GodModeOverlay } from "@/ui/godMode";
import { HelpOverlay } from "@/ui/helpOverlay";

// ============================================================================
// Scene Setup
// ============================================================================

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);
scene.fog = new THREE.FogExp2(0x0a0a1a, 0.00015); // low density for 10km world

// Camera — far plane 50km for massive chunked worlds
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 50000);
camera.position.set(150, 200, 150);
camera.lookAt(64, 0, 64);

// Renderer
const glRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
glRenderer.setSize(window.innerWidth, window.innerHeight);
glRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
glRenderer.shadowMap.enabled = true;
glRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
glRenderer.toneMapping = THREE.ACESFilmicToneMapping;
glRenderer.toneMappingExposure = 1.2;
document.getElementById("app")!.appendChild(glRenderer.domElement);

// Controls — orbit camera with full rotation
const controls = new OrbitControls(camera, glRenderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enableRotate = true;
controls.enablePan = true;
controls.enableZoom = true;
controls.maxPolarAngle = Math.PI * 0.49; // nearly overhead allowed
controls.minDistance = 10;
controls.maxDistance = 15000; // allow zooming way out for 10km world
controls.target.set(64, 0, 64);
controls.panSpeed = 3.0; // faster pan for large worlds
controls.zoomSpeed = 1.5;
controls.screenSpacePanning = false; // pan along ground plane, not screen plane
// Left-click drag = rotate, right-click drag = pan, scroll = zoom
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};

// ============================================================================
// Lighting
// ============================================================================

const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.8);
sunLight.position.set(5000, 8000, 3000);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(4096, 4096);
sunLight.shadow.camera.near = 1;
sunLight.shadow.camera.far = 20000;
sunLight.shadow.camera.left = -5000;
sunLight.shadow.camera.right = 5000;
sunLight.shadow.camera.top = 5000;
sunLight.shadow.camera.bottom = -5000;
scene.add(sunLight);

const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x362907, 0.5);
scene.add(hemiLight);

// ============================================================================
// Renderers
// ============================================================================

const terrainRenderer = new TerrainRenderer(scene);
const droneRenderer = new DroneRenderer(scene, 100);
const fogRenderer = new FogOfWarRenderer(scene);
const overlayRenderer = new OverlayRenderer(scene, 100, 50);

// ============================================================================
// State
// ============================================================================

let latestState: StateUpdate | null = null;

// Chunked world state
let worldSize = 0;
let firstChunkReceived = false;

// Hazard visualization
let hazardMeshes: THREE.Mesh[] = [];
let hazardsBuilt = false;

function buildHazardMeshes(hazards: HazardInfo[]): void {
  // Remove old hazard meshes
  for (const m of hazardMeshes) {
    scene.remove(m);
    m.geometry.dispose();
    (m.material as THREE.Material).dispose();
  }
  hazardMeshes = [];

  for (const h of hazards) {
    const isNoFly = h.type === "no_fly_zone";
    const color = isNoFly ? 0xff0000 : 0xaa00ff;
    const opacity = isNoFly ? 0.15 : 0.12;

    // Fill circle
    const circleGeo = new THREE.CircleGeometry(h.radius, 48);
    const circleMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const circleMesh = new THREE.Mesh(circleGeo, circleMat);
    circleMesh.rotation.x = -Math.PI / 2; // lay flat on XZ plane
    circleMesh.position.set(h.center[0], 0.5, h.center[2]); // slightly above ground
    scene.add(circleMesh);
    hazardMeshes.push(circleMesh);

    // Ring outline
    const ringGeo = new THREE.RingGeometry(h.radius - 0.5, h.radius, 48);
    const ringMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: opacity + 0.15,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.set(h.center[0], 0.6, h.center[2]);
    scene.add(ringMesh);
    hazardMeshes.push(ringMesh);
  }
}

function clearHazardMeshes(): void {
  for (const m of hazardMeshes) {
    scene.remove(m);
    m.geometry.dispose();
    (m.material as THREE.Material).dispose();
  }
  hazardMeshes = [];
  hazardsBuilt = false;
}

// ============================================================================
// HUD
// ============================================================================

const hudDroneCount = document.getElementById("drone-count")!;
const hudSurvivorCount = document.getElementById("survivor-count")!;
const hudCoverage = document.getElementById("coverage-pct")!;
const hudElapsed = document.getElementById("elapsed-time")!;
const hudPhase = document.getElementById("agent-phase")!;
const hudAiCalls = document.getElementById("ai-calls")!;
const hudWind = document.getElementById("wind-info")!;
const hudDaytime = document.getElementById("daytime-info")!;
const hudBriefing = document.getElementById("briefing")!;
const connStatus = document.getElementById("connection-status")!;
const connText = connStatus.querySelector("span")!;

const PHASE_LABELS: Record<string, string> = {
  INITIAL_SPREAD: "Spreading",
  SYSTEMATIC_SEARCH: "Searching",
  FRONTIER_HUNT: "Frontier Hunt",
  PRIORITY_SWEEP: "Priority Sweep",
  COMPLETE: "Complete",
};

function updateHUD(state: StateUpdate): void {
  const active = state.drones.filter((d) => d.status === "active").length;
  const total = state.drones.length;
  hudDroneCount.textContent = `${active}/${total}`;

  const found = state.survivors.length;
  hudSurvivorCount.textContent = `${found}`;

  hudCoverage.textContent = `${state.coverage_pct}%`;

  const mins = Math.floor(state.elapsed / 60);
  const secs = Math.floor(state.elapsed % 60);
  hudElapsed.textContent = `${mins}:${secs.toString().padStart(2, "0")}`;

  if (state.agent_info) {
    hudPhase.textContent = PHASE_LABELS[state.agent_info.phase] ?? state.agent_info.phase;
    hudAiCalls.textContent = `${state.agent_info.planner_calls} / ${state.agent_info.reasoner_calls}`;

    if (state.agent_info.briefing) {
      hudBriefing.style.display = "block";
      hudBriefing.textContent = state.agent_info.briefing;
    }

    // Weather HUD
    if (state.agent_info.weather) {
      const w = state.agent_info.weather;
      const dir = ((w.wind_direction * 180) / Math.PI).toFixed(0);
      hudWind.textContent = `${w.wind_speed.toFixed(1)} m/s ${dir}°${w.gusting ? " GUST" : ""}`;
    }

    // Day/night cycle — update lighting
    if (state.agent_info.daycycle) {
      const dc = state.agent_info.daycycle;
      const DAY_LABELS: Record<string, string> = {
        dawn: "Dawn",
        day: "Day",
        dusk: "Dusk",
        night: "Night",
      };
      hudDaytime.textContent = DAY_LABELS[dc.phase] ?? dc.phase;

      // Update sun light
      sunLight.intensity = 0.3 + dc.sun_intensity * 1.5;
      sunLight.color.setRGB(dc.sun_color[0], dc.sun_color[1], dc.sun_color[2]);

      // Update ambient
      ambientLight.intensity = 0.15 + dc.sun_intensity * 0.45;

      // Update scene background (darker at night)
      const bg = 0.02 + dc.sun_intensity * 0.06;
      scene.background = new THREE.Color(bg, bg, bg + 0.02);

      // Update fog density (denser at night for atmosphere)
      (scene.fog as THREE.FogExp2).density = 0.00015 + (1 - dc.sun_intensity) * 0.0002;
    }

    // Feed activity log entries
    if (state.agent_info.activity_log && state.agent_info.activity_log.length > 0) {
      activityLog.addEntries(state.agent_info.activity_log);
    }
  }
}

function setConnected(connected: boolean): void {
  if (connected) {
    connStatus.classList.add("connected");
    connText.textContent = "Connected";
  } else {
    connStatus.classList.remove("connected");
    connText.textContent = "Disconnected";
  }
}

// ============================================================================
// WebSocket
// ============================================================================

const client = new SwarmClient(
  "ws://localhost:8765/ws",
  (state: StateUpdate) => {
    // Build/rebuild monolithic terrain when it arrives (legacy small maps)
    if (state.terrain) {
      const decoded = decodeTerrain(state.terrain);
      terrainRenderer.buildFromData(decoded);
      fogRenderer.initialize(decoded.width, decoded.height);
      clearHazardMeshes();

      const cx = decoded.width / 2;
      const cz = decoded.height / 2;
      const camDist = Math.max(decoded.width, decoded.height) * 0.6;
      controls.target.set(cx, 30, cz);
      camera.position.set(cx + camDist * 0.7, camDist * 0.8, cz + camDist * 0.7);
      controls.maxDistance = camDist * 3;
      // monolithic mode

      console.log("[DroneSwarm] Monolithic terrain built");
    }

    // Detect chunked mode from state fields
    const rawState = state as unknown as Record<string, unknown>;
    if (rawState.world_size && rawState.chunk_size) {
      worldSize = rawState.world_size as number;
    }

    // Log survivor data on first few ticks
    if (state.tick <= 3) {
      const raw = state as unknown as Record<string, unknown>;
      const allS = raw.all_survivors as unknown[];
      console.log(`[DroneSwarm] tick=${state.tick} all_survivors=${allS?.length ?? 'MISSING'} survivors=${state.survivors.length}`);
      if (allS && allS.length > 0) {
        console.log(`[DroneSwarm] first survivor:`, allS[0]);
      }
    }

    // Center camera on drone fleet if not already done
    centerCameraOnDrones();

    // Build hazard meshes once when data first arrives
    if (!hazardsBuilt && state.agent_info?.hazards && state.agent_info.hazards.length > 0) {
      buildHazardMeshes(state.agent_info.hazards);
      hazardsBuilt = true;
    }

    latestState = state;
  },
  setConnected,
);

// Track whether camera has been centered on the drone fleet
let cameraCentered = false;

function centerCameraOnDrones(): void {
  if (cameraCentered || !latestState || latestState.drones.length === 0) return;
  cameraCentered = true;
  const d = latestState.drones[0];
  const cx = d.position[0];
  const cz = d.position[2];
  const camDist = 2000;
  controls.target.set(cx, 30, cz);
  camera.position.set(cx + camDist * 0.4, camDist * 0.5, cz + camDist * 0.4);
  controls.maxDistance = worldSize * 1.5 || 15000;
  console.log(`[DroneSwarm] Camera centered on drone fleet at (${cx.toFixed(0)}, ${cz.toFixed(0)})`);
}

// Handle chunk terrain messages
client.onChunkTerrain((chunk: ChunkTerrainData) => {
  terrainRenderer.buildChunk(chunk);

  if (!firstChunkReceived) {
    firstChunkReceived = true;
    controls.maxDistance = worldSize * 1.5 || chunk.size * 10;
  }

  // Try to center camera once we have drone positions
  centerCameraOnDrones();

  // Initialize fog-of-war plane once we know the world size.
  // Fog grid resolution comes from the state_update fog_grid dimensions (separate from world size).
  if (worldSize > 0 && !fogRenderer.isInitialized() && latestState?.fog_grid) {
    const fogW = latestState.fog_grid.width;
    const fogH = latestState.fog_grid.height;
    fogRenderer.initialize(worldSize, worldSize, fogW, fogH);
    console.log(`[DroneSwarm] Fog-of-war initialized: ${worldSize}m world, ${fogW}x${fogH} grid`);
  }

  console.log(
    `[DroneSwarm] Chunk (${chunk.cx},${chunk.cz}) built — ${terrainRenderer.getChunkCount()} chunks loaded`,
  );
});

// Handle world overview for minimap
client.onWorldOverview((overview: WorldOverview) => {
  worldSize = overview.world_size;
  minimap.setOverview(overview);
  console.log(
    `[DroneSwarm] World: ${overview.world_size}m, ${overview.chunks_x}x${overview.chunks_z} chunks`,
  );
});

// ============================================================================
// Interaction + Chat
// ============================================================================

const interaction = new InteractionManager(scene, camera, glRenderer, client);
const minimap = new Minimap();
const chatPanel = new ChatPanel(client);
const activityLog = new ActivityLog();
const godMode = new GodModeOverlay(scene);
const helpOverlay = new HelpOverlay();
const settingsPanel = new SettingsPanel((config: SimSettings) => {
  // Send reset with config to backend
  paused = false;
  document.getElementById("btn-pause")!.textContent = "Pause";
  // Clear all chunk state on reset
  firstChunkReceived = false;
  cameraCentered = false;
  terrainRenderer.dispose();
  fogRenderer.dispose();
  clearHazardMeshes();
  client.sendSimControl("reset", undefined, config as unknown as Record<string, number>);
  settingsPanel.toggle();
  console.log("[DroneSwarm] Reset with custom config:", config);
});

// Wire chat responses from server to the chat panel
client.onChatResponse((msg) => chatPanel.handleResponse(msg));
client.connect();

// ============================================================================
// Render Loop
// ============================================================================

const clock = new THREE.Clock();

// WASD pan state
const panKeys = { w: false, a: false, s: false, d: false };
window.addEventListener("keydown", (e: KeyboardEvent) => {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  if (e.key === "w") panKeys.w = true;
  if (e.key === "a") panKeys.a = true;
  if (e.key === "s") panKeys.s = true;
  if (e.key === "d") panKeys.d = true;
});
window.addEventListener("keyup", (e: KeyboardEvent) => {
  if (e.key === "w") panKeys.w = false;
  if (e.key === "a") panKeys.a = false;
  if (e.key === "s") panKeys.s = false;
  if (e.key === "d") panKeys.d = false;
});

let lastFrameTime = 0;
const FRAME_INTERVAL = 1000 / 30; // Cap at 30fps to reduce CPU/GPU load

function animate(now: number = 0): void {
  requestAnimationFrame(animate);

  // Throttle to 30fps
  if (now - lastFrameTime < FRAME_INTERVAL) return;
  lastFrameTime = now;

  const dt = clock.getDelta();

  // WASD panning — move camera target along ground plane
  const panSpeed = 800 * dt; // meters per second
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
  const panDelta = new THREE.Vector3();
  if (panKeys.w) panDelta.add(forward.clone().multiplyScalar(panSpeed));
  if (panKeys.s) panDelta.add(forward.clone().multiplyScalar(-panSpeed));
  if (panKeys.d) panDelta.add(right.clone().multiplyScalar(panSpeed));
  if (panKeys.a) panDelta.add(right.clone().multiplyScalar(-panSpeed));
  if (panDelta.lengthSq() > 0) {
    controls.target.add(panDelta);
    camera.position.add(panDelta);
  }

  // Update renderers from latest state
  if (latestState) {
    droneRenderer.update(latestState.drones, dt);
    if (latestState.fog_grid) {
      // Lazy-init fog renderer once we have both world size and fog grid dimensions
      if (worldSize > 0 && !fogRenderer.isInitialized()) {
        fogRenderer.initialize(worldSize, worldSize, latestState.fog_grid.width, latestState.fog_grid.height);
      }
      fogRenderer.updateFromRLE(latestState.fog_grid);
    }
    overlayRenderer.update(latestState.drones, latestState.comms_links, latestState.survivors, dt);
    // Only show god mode survivors on loaded terrain chunks
    const chunkSz = (latestState as unknown as Record<string, unknown>).chunk_size as number | undefined;
    const visibleSurvivors = chunkSz && latestState.all_survivors
      ? latestState.all_survivors.filter((s) =>
          terrainRenderer.hasChunk(
            Math.floor(s.position[0] / chunkSz),
            Math.floor(s.position[2] / chunkSz),
          ),
        )
      : latestState.all_survivors;
    godMode.update(visibleSurvivors, dt);
    minimap.update(latestState.drones, latestState.all_survivors);
    interaction.update(latestState.drones);
    updateHUD(latestState);
  }

  controls.update();
  glRenderer.render(scene, camera);
}

animate();

// ============================================================================
// Resize
// ============================================================================

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  glRenderer.setSize(window.innerWidth, window.innerHeight);
});

// ============================================================================
// Controls
// ============================================================================

let paused = false;

function togglePause(): void {
  paused = !paused;
  client.sendSimControl(paused ? "pause" : "resume");
  const btn = document.getElementById("btn-pause")!;
  btn.textContent = paused ? "Resume" : "Pause";
}

function resetSim(): void {
  paused = false;
  document.getElementById("btn-pause")!.textContent = "Pause";
  // Clear ALL visual state so user sees the reset
  latestState = null;
  firstChunkReceived = false;
  cameraCentered = false;
  clearHazardMeshes();
  terrainRenderer.dispose();
  fogRenderer.dispose();
  client.sendSimControl("reset");
  console.log("[DroneSwarm] Reset requested — clearing scene, new terrain incoming");
}

// Button listeners
document
  .getElementById("btn-pause")!
  .addEventListener("click", togglePause);
document
  .getElementById("btn-speed1")!
  .addEventListener("click", () => client.sendSimControl("set_speed", 1.0));
document
  .getElementById("btn-speed2")!
  .addEventListener("click", () => client.sendSimControl("set_speed", 2.0));
document
  .getElementById("btn-speed5")!
  .addEventListener("click", () => client.sendSimControl("set_speed", 5.0));
document.getElementById("btn-reset")!.addEventListener("click", resetSim);

// Keyboard shortcuts
window.addEventListener("keydown", (e: KeyboardEvent) => {
  // Don't handle shortcuts when chat or settings inputs are focused
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
    return;
  }

  if (e.key === " ") {
    e.preventDefault();
    togglePause();
  } else if (e.key === "r" || e.key === "R") {
    client.sendSimControl("resume");
    paused = false;
    document.getElementById("btn-pause")!.textContent = "Pause";
  } else if (e.key === "1") {
    client.sendSimControl("set_speed", 1.0);
  } else if (e.key === "2") {
    client.sendSimControl("set_speed", 2.0);
  } else if (e.key === "3") {
    client.sendSimControl("set_speed", 5.0);
  } else if (e.key === "n" || e.key === "N") {
    resetSim();
  } else if (e.key === "t" || e.key === "T") {
    chatPanel.toggle();
  } else if (e.key === "g" || e.key === "G") {
    godMode.toggle();
    fogRenderer.setVisible(!godMode.isActive());
  } else if (e.key === "?" || e.key === "F1") {
    e.preventDefault();
    helpOverlay.toggle();
  }
});

// Expose debug hooks for visual testing
(window as unknown as Record<string, unknown>).__DEBUG = {
  camera,
  controls,
  scene,
  godMode,
  getState: () => latestState,
};

console.log("[DroneSwarm] Frontend initialized — connecting to backend...");
console.log("[DroneSwarm] Controls: Space=pause, 1/2/3=speed, N=new sim, T=chat");
console.log("[DroneSwarm] Click drone to select, Right-click terrain to send drone");
