/**
 * WebSocket client for connecting to the drone swarm backend.
 * Auto-reconnects with exponential backoff.
 */

import type { StateUpdate, TerrainData } from "./types";

export type StateCallback = (state: StateUpdate) => void;
export type ConnectionCallback = (connected: boolean) => void;
export type ChatCallback = (message: string) => void;

export class SwarmClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 16000;
  private onState: StateCallback;
  private onConnection: ConnectionCallback;
  private onChat: ChatCallback | null = null;
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

  sendSimControl(action: "pause" | "resume" | "set_speed" | "reset", value?: number): void {
    this.send({
      type: "sim_control",
      action,
      value: value ?? null,
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
