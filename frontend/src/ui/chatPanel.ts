/**
 * Chat panel for natural language commands to the drone swarm.
 * Overlays the bottom-right of the screen.
 */

import type { SwarmClient } from "@/network/client";

const MAX_MESSAGES = 50;

interface ChatMessage {
  role: "user" | "system";
  text: string;
  timestamp: number;
}

export class ChatPanel {
  private container: HTMLDivElement;
  private messagesDiv: HTMLDivElement;
  private input: HTMLInputElement;
  private client: SwarmClient;
  private messages: ChatMessage[] = [];
  private visible = false;

  constructor(client: SwarmClient) {
    this.client = client;

    // Build DOM
    this.container = document.createElement("div");
    this.container.id = "chat-panel";
    Object.assign(this.container.style, {
      position: "absolute",
      bottom: "60px",
      right: "16px",
      width: "340px",
      maxHeight: "400px",
      background: "rgba(0, 0, 0, 0.8)",
      border: "1px solid rgba(0, 200, 255, 0.3)",
      borderRadius: "8px",
      display: "none",
      flexDirection: "column",
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      fontSize: "13px",
      backdropFilter: "blur(8px)",
      overflow: "hidden",
    });

    // Header
    const header = document.createElement("div");
    Object.assign(header.style, {
      padding: "8px 12px",
      borderBottom: "1px solid rgba(0, 200, 255, 0.2)",
      color: "#00c8ff",
      fontSize: "12px",
      letterSpacing: "1px",
      textTransform: "uppercase",
      fontWeight: "600",
    });
    header.textContent = "Mission Command";
    this.container.appendChild(header);

    // Messages area
    this.messagesDiv = document.createElement("div");
    Object.assign(this.messagesDiv.style, {
      flex: "1",
      overflowY: "auto",
      padding: "8px 12px",
      maxHeight: "300px",
    });
    this.container.appendChild(this.messagesDiv);

    // Input area
    const inputRow = document.createElement("div");
    Object.assign(inputRow.style, {
      display: "flex",
      borderTop: "1px solid rgba(0, 200, 255, 0.2)",
    });

    this.input = document.createElement("input");
    this.input.type = "text";
    this.input.placeholder = "Command the swarm...";
    Object.assign(this.input.style, {
      flex: "1",
      background: "rgba(0, 0, 0, 0.5)",
      border: "none",
      color: "#e0e0e0",
      padding: "10px 12px",
      fontSize: "13px",
      outline: "none",
    });

    this.input.addEventListener("keydown", (e: KeyboardEvent) => {
      e.stopPropagation(); // Don't trigger scene keyboard shortcuts
      if (e.key === "Enter" && this.input.value.trim()) {
        this.sendMessage(this.input.value.trim());
        this.input.value = "";
      } else if (e.key === "Escape") {
        this.toggle();
      }
    });

    inputRow.appendChild(this.input);
    this.container.appendChild(inputRow);

    document.getElementById("app")!.appendChild(this.container);

    // Add initial system message
    this.addSystemMessage(
      'Mission Command ready. Try "status", "focus search north", or "pull back drone 3".',
    );
  }

  toggle(): void {
    this.visible = !this.visible;
    this.container.style.display = this.visible ? "flex" : "none";
    if (this.visible) {
      this.input.focus();
    }
  }

  isVisible(): boolean {
    return this.visible;
  }

  /** Handle a chat_response message from the server. */
  handleResponse(message: string): void {
    this.addSystemMessage(message);
  }

  private sendMessage(text: string): void {
    this.addMessage({ role: "user", text, timestamp: Date.now() });
    this.client.sendChat(text);
    // Response will come back via WebSocket as a chat_response message
  }

  private addSystemMessage(text: string): void {
    this.addMessage({ role: "system", text, timestamp: Date.now() });
  }

  private addMessage(msg: ChatMessage): void {
    this.messages.push(msg);
    if (this.messages.length > MAX_MESSAGES) {
      this.messages.shift();
    }
    this.renderMessages();
  }

  private renderMessages(): void {
    // Clear and re-render (simple approach, fine for <50 messages)
    this.messagesDiv.innerHTML = "";
    for (const msg of this.messages) {
      const el = document.createElement("div");
      Object.assign(el.style, {
        marginBottom: "6px",
        lineHeight: "1.4",
        wordWrap: "break-word",
      });

      if (msg.role === "user") {
        el.innerHTML = `<span style="color:#ffaa00;font-weight:600">Operator:</span> <span style="color:#e0e0e0">${this.escapeHtml(msg.text)}</span>`;
      } else {
        el.innerHTML = `<span style="color:#44cc88;font-weight:600">AI:</span> <span style="color:#b0b0b0">${this.escapeHtml(msg.text)}</span>`;
      }

      this.messagesDiv.appendChild(el);
    }

    // Auto-scroll to bottom
    this.messagesDiv.scrollTop = this.messagesDiv.scrollHeight;
  }

  private escapeHtml(text: string): string {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  dispose(): void {
    this.container.remove();
  }
}
