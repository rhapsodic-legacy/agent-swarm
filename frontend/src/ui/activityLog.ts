/**
 * Activity log panel — shows a scrolling feed of drone decisions, events, and alerts.
 * Toggle visibility with the "L" key.
 */

import type { ActivityLogEntry } from "@/network/types";

const MAX_DISPLAY_ENTRIES = 80;

const CATEGORY_COLORS: Record<string, string> = {
  info: "#888",
  decision: "#00c8ff",
  alert: "#ff6644",
  event: "#44cc88",
};

const CATEGORY_LABELS: Record<string, string> = {
  info: "INFO",
  decision: "AI",
  alert: "ALERT",
  event: "EVENT",
};

export class ActivityLog {
  private container: HTMLDivElement;
  private logBody!: HTMLDivElement;
  private visible = true;
  private entries: ActivityLogEntry[] = [];
  private seenTicks = new Set<string>(); // dedup key: tick+drone_id+message
  private autoScroll = true;

  constructor() {
    this.container = this.buildPanel();
    document.getElementById("app")!.appendChild(this.container);

    window.addEventListener("keydown", (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "l" || e.key === "L") {
        this.toggle();
      }
    });
  }

  toggle(): void {
    this.visible = !this.visible;
    this.container.style.display = this.visible ? "flex" : "none";
  }

  /** Feed new log entries from the state update. Deduplicates automatically. */
  addEntries(entries: ActivityLogEntry[]): void {
    for (const entry of entries) {
      const key = `${entry.tick}:${entry.drone_id}:${entry.message}`;
      if (this.seenTicks.has(key)) continue;
      this.seenTicks.add(key);
      this.entries.push(entry);
    }

    // Trim old entries
    if (this.entries.length > MAX_DISPLAY_ENTRIES * 2) {
      this.entries = this.entries.slice(-MAX_DISPLAY_ENTRIES);
      // Rebuild seen set from remaining entries
      this.seenTicks.clear();
      for (const e of this.entries) {
        this.seenTicks.add(`${e.tick}:${e.drone_id}:${e.message}`);
      }
    }

    this.render();
  }

  private render(): void {
    const display = this.entries.slice(-MAX_DISPLAY_ENTRIES);

    // Build HTML in one pass
    let html = "";
    for (const entry of display) {
      const color = CATEGORY_COLORS[entry.category] ?? "#888";
      const label = CATEGORY_LABELS[entry.category] ?? "LOG";
      const mins = Math.floor(entry.elapsed / 60);
      const secs = Math.floor(entry.elapsed % 60);
      const time = `${mins}:${secs.toString().padStart(2, "0")}`;
      const drone = entry.drone_id !== null ? `D${entry.drone_id}` : "MCP";
      const droneColor = entry.drone_id !== null ? "#aaa" : "#00c8ff";

      html += `<div style="margin-bottom:3px;line-height:1.35;font-size:12px;">`;
      html += `<span style="color:#555">${time}</span> `;
      html += `<span style="color:${color};font-weight:600;font-size:10px">[${label}]</span> `;
      html += `<span style="color:${droneColor}">${drone}</span> `;
      html += `<span style="color:#ccc">${this.escapeHtml(entry.message)}</span>`;
      html += `</div>`;
    }

    this.logBody.innerHTML = html;

    // Auto-scroll to bottom
    if (this.autoScroll) {
      this.logBody.scrollTop = this.logBody.scrollHeight;
    }
  }

  private buildPanel(): HTMLDivElement {
    const panel = document.createElement("div");
    Object.assign(panel.style, {
      position: "absolute",
      top: "240px",
      left: "16px",
      width: "340px",
      maxHeight: "400px",
      background: "rgba(0, 0, 0, 0.75)",
      border: "1px solid rgba(0, 200, 255, 0.25)",
      borderRadius: "8px",
      display: "flex",
      flexDirection: "column",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
      fontSize: "12px",
      backdropFilter: "blur(8px)",
      overflow: "hidden",
      pointerEvents: "auto",
      zIndex: "50",
    });

    // Header
    const header = document.createElement("div");
    Object.assign(header.style, {
      padding: "6px 10px",
      borderBottom: "1px solid rgba(0, 200, 255, 0.2)",
      color: "#00c8ff",
      fontSize: "11px",
      letterSpacing: "1px",
      textTransform: "uppercase",
      fontWeight: "600",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      flexShrink: "0",
    });
    header.innerHTML = `<span>Activity Log</span><span style="color:#555;font-size:10px">L to toggle</span>`;
    panel.appendChild(header);

    // Scrollable log body
    this.logBody = document.createElement("div");
    Object.assign(this.logBody.style, {
      flex: "1",
      overflowY: "auto",
      overflowX: "hidden",
      padding: "6px 10px",
      maxHeight: "360px",
    });

    // Detect manual scroll to pause auto-scroll
    this.logBody.addEventListener("scroll", () => {
      const { scrollTop, scrollHeight, clientHeight } = this.logBody;
      this.autoScroll = scrollHeight - scrollTop - clientHeight < 30;
    });

    panel.appendChild(this.logBody);
    return panel;
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
