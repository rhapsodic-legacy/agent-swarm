/**
 * WebSocket client for connecting to the drone swarm backend.
 * Auto-reconnects with exponential backoff.
 * Supports both monolithic terrain and chunked world messages.
 */

import type { StateUpdate, TerrainData, ChunkTerrainData, WorldOverview } from "./types";

export type StateCallback = (state: StateUpdate) => void;
export type ConnectionCallback = (connected: boolean) => void;
export type ChatCallback = (message: string) => void;
export type ChunkCallback = (chunk: ChunkTerrainData) => void;
export type OverviewCallback = (overview: WorldOverview) => void;

export class SwarmClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 16000;
  private onState: StateCallback;
  private onConnection: ConnectionCallback;
  private onChat: ChatCallback | null = null;
  private onChunk: ChunkCallback | null = null;
  private onOverview: OverviewCallback | null = null;
  private terrain: TerrainData | null = null;
  private latestState: StateUpdate | null = null;
  private shouldReconnect = true;

  constructor(url: string, onState: StateCallback, onConnection: ConnectionCallback) {
    this.url = url;
    this.onState = onState;
    this.onConnection = onConnection;
  }

  connect(): void {
    this.shouldReconnect = true;
    this.attemptConnect();
  }

  disconnect(): void {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendCommand(command: string, droneId: number, target?: [number, number]): void {
    this.send({
      type: "command",
      command,
      drone_id: droneId,
      target: target ?? null,
    });
  }

  sendSimControl(
    action: "pause" | "resume" | "set_speed" | "reset",
    value?: number,
    config?: Record<string, number>,
  ): void {
    this.send({
      type: "sim_control",
      action,
      value: value ?? null,
      config: config ?? null,
    });
  }

  sendChat(message: string): void {
    this.send({
      type: "chat_message",
      message,
    });
  }

  onChatResponse(callback: ChatCallback): void {
    this.onChat = callback;
  }

  onChunkTerrain(callback: ChunkCallback): void {
    this.onChunk = callback;
  }

  onWorldOverview(callback: OverviewCallback): void {
    this.onOverview = callback;
  }

  getTerrain(): TerrainData | null {
    return this.terrain;
  }

  getLatestState(): StateUpdate | null {
    return this.latestState;
  }

  private send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private attemptConnect(): void {
    if (!this.shouldReconnect) return;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log("[SwarmClient] Connected");
        this.reconnectDelay = 1000;
        this.onConnection(true);
      };

      this.ws.onmessage = (event: MessageEvent) => {
        const raw = JSON.parse(event.data as string) as Record<string, unknown>;
        const msgType = raw.type as string;

        if (msgType === "state_update") {
          const msg = raw as unknown as StateUpdate;
          if (msg.terrain) {
            this.terrain = msg.terrain;
            console.log(
              `[SwarmClient] Terrain received: ${msg.terrain.width}x${msg.terrain.height}`,
            );
          }
          this.latestState = msg;
          this.onState(msg);
        } else if (msgType === "chunk_terrain") {
          if (this.onChunk) {
            const chunk = raw as unknown as import("./types").ChunkTerrainData;
            console.log(
              `[SwarmClient] Chunk received: (${chunk.cx},${chunk.cz}) at (${chunk.origin_x},${chunk.origin_z})`,
            );
            this.onChunk(chunk);
          }
        } else if (msgType === "world_overview") {
          if (this.onOverview) {
            const overview = raw as unknown as import("./types").WorldOverview;
            console.log(
              `[SwarmClient] World overview: ${overview.world_size}m, ${overview.chunks_x}x${overview.chunks_z} chunks`,
            );
            this.onOverview(overview);
          }
        } else if (msgType === "chat_response") {
          const chatMsg = raw.message as string | undefined;
          if (chatMsg && this.onChat) {
            this.onChat(chatMsg);
          }
        }
      };

      this.ws.onclose = () => {
        console.log("[SwarmClient] Disconnected");
        this.onConnection(false);
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        // onclose will fire after onerror, which handles reconnection
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect) return;
    console.log(`[SwarmClient] Reconnecting in ${this.reconnectDelay}ms...`);
    setTimeout(() => this.attemptConnect(), this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}
