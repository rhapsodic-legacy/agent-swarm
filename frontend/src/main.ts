import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SwarmClient } from "@/network/client";
import type { StateUpdate } from "@/network/types";
import { TerrainRenderer } from "@/scene/terrain";
import { DroneRenderer } from "@/entities/drones";
import { FogOfWarRenderer } from "@/fog/fogOfWar";
import { OverlayRenderer } from "@/entities/overlays";
import { InteractionManager } from "@/ui/interaction";
import { ChatPanel } from "@/ui/chatPanel";
import { SettingsPanel } from "@/ui/settingsPanel";
import type { SimSettings } from "@/ui/settingsPanel";

// ============================================================================
// Scene Setup
// ============================================================================

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);
scene.fog = new THREE.FogExp2(0x0a0a1a, 0.0008);

// Camera
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(150, 200, 150);
camera.lookAt(64, 0, 64);

// Renderer
const glRenderer = new THREE.WebGLRenderer({ antialias: true });
glRenderer.setSize(window.innerWidth, window.innerHeight);
glRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
glRenderer.shadowMap.enabled = true;
glRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
glRenderer.toneMapping = THREE.ACESFilmicToneMapping;
glRenderer.toneMappingExposure = 1.2;
document.getElementById("app")!.appendChild(glRenderer.domElement);

// Controls
const controls = new OrbitControls(camera, glRenderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI * 0.45;
controls.minDistance = 10;
controls.maxDistance = 800;
controls.target.set(64, 0, 64);

// ============================================================================
// Lighting
// ============================================================================

const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.8);
sunLight.position.set(200, 300, 100);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.near = 0.5;
sunLight.shadow.camera.far = 800;
sunLight.shadow.camera.left = -200;
sunLight.shadow.camera.right = 200;
sunLight.shadow.camera.top = 200;
sunLight.shadow.camera.bottom = -200;
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
      (scene.fog as THREE.FogExp2).density = 0.0008 + (1 - dc.sun_intensity) * 0.001;
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
    // Build/rebuild terrain when it arrives (initial load or after reset)
    if (state.terrain) {
      terrainRenderer.buildFromData(state.terrain);
      fogRenderer.initialize(state.terrain.width, state.terrain.height);

      // Re-center camera on terrain
      const cx = state.terrain.width / 2;
      const cz = state.terrain.height / 2;
      controls.target.set(cx, 30, cz);
      camera.position.set(cx + 100, 180, cz + 100);

      console.log("[DroneSwarm] Terrain built, visualization active");
    }

    latestState = state;
  },
  setConnected,
);

// ============================================================================
// Interaction + Chat
// ============================================================================

const interaction = new InteractionManager(scene, camera, glRenderer, client);
const chatPanel = new ChatPanel(client);
const settingsPanel = new SettingsPanel((config: SimSettings) => {
  // Send reset with config to backend
  paused = false;
  document.getElementById("btn-pause")!.textContent = "Pause";
  client.sendSimControl("reset", undefined, config as unknown as Record<string, number>);
  settingsPanel.toggle(); // Close panel after apply
  console.log("[DroneSwarm] Reset with custom config:", config);
});

// Wire chat responses from server to the chat panel
client.onChatResponse((msg) => chatPanel.handleResponse(msg));
client.connect();

// ============================================================================
// Render Loop
// ============================================================================

const clock = new THREE.Clock();

function animate(): void {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();

  // Update renderers from latest state
  if (latestState) {
    droneRenderer.update(latestState.drones, dt);
    fogRenderer.updateFromRLE(latestState.fog_grid);
    overlayRenderer.update(latestState.drones, latestState.comms_links, latestState.survivors, dt);
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
  client.sendSimControl("reset");
  console.log("[DroneSwarm] Reset requested — new terrain incoming");
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
  }
});

console.log("[DroneSwarm] Frontend initialized — connecting to backend...");
console.log("[DroneSwarm] Controls: Space=pause, 1/2/3=speed, N=new sim, T=chat");
console.log("[DroneSwarm] Click drone to select, Right-click terrain to send drone");
