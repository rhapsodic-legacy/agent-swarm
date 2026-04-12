/**
 * Minimap — shows a top-down overview of the 10km world.
 * 1 pixel per chunk from the backend overview, scaled up.
 * Drone positions drawn as dots. Survivor clusters as red marks.
 */

import type { DroneState, SurvivorState, WorldOverview } from "@/network/types";

const MAP_SIZE = 180; // pixels on screen
const BORDER_COLOR = "rgba(0, 200, 255, 0.4)";
const DRONE_COLOR = "#00ffff";
const SURVIVOR_COLOR = "#ff2200";
const BASE_COLOR = "#ffaa00";

export class Minimap {
  private container: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private overview: WorldOverview | null = null;
  private overviewImage: ImageData | null = null;
  private worldSize = 0;

  constructor() {
    this.container = document.createElement("div");
    Object.assign(this.container.style, {
      position: "absolute",
      bottom: "60px",
      right: "16px",
      width: `${MAP_SIZE}px`,
      height: `${MAP_SIZE}px`,
      border: `1px solid ${BORDER_COLOR}`,
      borderRadius: "4px",
      background: "rgba(0, 0, 0, 0.7)",
      backdropFilter: "blur(4px)",
      overflow: "hidden",
      zIndex: "150",
    });

    this.canvas = document.createElement("canvas");
    this.canvas.width = MAP_SIZE;
    this.canvas.height = MAP_SIZE;
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.container.appendChild(this.canvas);

    this.ctx = this.canvas.getContext("2d")!;
    this.ctx.imageSmoothingEnabled = false;

    document.getElementById("app")!.appendChild(this.container);
  }

  setOverview(overview: WorldOverview): void {
    this.overview = overview;
    this.worldSize = overview.world_size;

    // Decode the overview RGB data
    const bytes = Uint8Array.from(atob(overview.overview_rgb_b64), (c) => c.charCodeAt(0));
    const chunksX = overview.chunks_x;
    const chunksZ = overview.chunks_z;

    // Create a small ImageData from the overview (chunksX x chunksZ)
    const imgData = new ImageData(chunksX, chunksZ);
    for (let i = 0; i < chunksX * chunksZ; i++) {
      imgData.data[i * 4] = bytes[i * 3];     // R
      imgData.data[i * 4 + 1] = bytes[i * 3 + 1]; // G
      imgData.data[i * 4 + 2] = bytes[i * 3 + 2]; // B
      imgData.data[i * 4 + 3] = 255;           // A
    }
    this.overviewImage = imgData;
  }

  update(drones: DroneState[], survivors?: SurvivorState[]): void {
    if (!this.overview || !this.overviewImage) return;

    const ctx = this.ctx;
    const ws = this.worldSize;

    // Draw overview terrain (scaled up to fill the minimap)
    // Create a temp canvas to hold the small image, then drawImage with scaling
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = this.overview.chunks_x;
    tmpCanvas.height = this.overview.chunks_z;
    tmpCanvas.getContext("2d")!.putImageData(this.overviewImage, 0, 0);

    ctx.clearRect(0, 0, MAP_SIZE, MAP_SIZE);
    ctx.drawImage(tmpCanvas, 0, 0, MAP_SIZE, MAP_SIZE);

    // Darken slightly for contrast
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    ctx.fillRect(0, 0, MAP_SIZE, MAP_SIZE);

    // Draw base position
    const baseX = (ws * 0.15 / ws) * MAP_SIZE;
    const baseZ = (ws * 0.15 / ws) * MAP_SIZE;
    ctx.fillStyle = BASE_COLOR;
    ctx.fillRect(baseX - 3, baseZ - 3, 6, 6);

    // Draw survivors
    if (survivors) {
      ctx.fillStyle = SURVIVOR_COLOR;
      for (const s of survivors) {
        const sx = (s.position[0] / ws) * MAP_SIZE;
        const sz = (s.position[2] / ws) * MAP_SIZE;
        ctx.fillRect(sx - 1, sz - 1, 3, 3);
      }
    }

    // Draw drones
    ctx.fillStyle = DRONE_COLOR;
    for (const d of drones) {
      const dx = (d.position[0] / ws) * MAP_SIZE;
      const dz = (d.position[2] / ws) * MAP_SIZE;
      ctx.beginPath();
      ctx.arc(dx, dz, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  dispose(): void {
    this.container.remove();
  }
}
