/**
 * Intel briefing overlay — shows scenario context (title, description, known
 * facts) when a new SAR mission starts. Driven by the `mission_briefing`
 * scenario message sent on connect and after each reset.
 *
 * Dismissable with Escape or the Close button. Auto-shows on every new
 * briefing so the operator sees the scenario change after a reset.
 */

import type { MissionScenario } from "@/network/types";

const STYLE_ID = "intel-briefing-styles";

export class IntelBriefing {
  private container: HTMLDivElement;
  private titleEl: HTMLDivElement;
  private descriptionEl: HTMLDivElement;
  private factsEl: HTMLUListElement;
  private survivalEl: HTMLDivElement;
  private visible = false;
  private current: MissionScenario | null = null;
  private keydownHandler: (e: KeyboardEvent) => void;

  constructor() {
    this.injectStyles();
    this.container = this.buildContainer();
    this.titleEl = this.container.querySelector(".intel-title")!;
    this.descriptionEl = this.container.querySelector(".intel-description")!;
    this.factsEl = this.container.querySelector(".intel-facts")!;
    this.survivalEl = this.container.querySelector(".intel-survival")!;
    document.getElementById("app")!.appendChild(this.container);

    this.keydownHandler = (e: KeyboardEvent) => this.handleKeydown(e);
    window.addEventListener("keydown", this.keydownHandler);
  }

  setBriefing(mission: MissionScenario): void {
    this.current = mission;
    this.titleEl.textContent = mission.title;
    this.descriptionEl.textContent = mission.description;

    this.factsEl.innerHTML = "";
    for (const fact of mission.known_facts) {
      const li = document.createElement("li");
      li.textContent = fact;
      this.factsEl.appendChild(li);
    }

    const minutes = Math.round(mission.survival_window_seconds / 60);
    const hours = minutes / 60;
    const windowText =
      hours >= 1 ? `${hours.toFixed(1)} hr` : `${minutes} min`;
    this.survivalEl.textContent = `Survival window: ${windowText}`;

    this.show();
  }

  show(): void {
    this.visible = true;
    this.container.style.display = "flex";
  }

  hide(): void {
    this.visible = false;
    this.container.style.display = "none";
  }

  toggle(): void {
    if (!this.current) return;
    if (this.visible) {
      this.hide();
    } else {
      this.show();
    }
  }

  isVisible(): boolean {
    return this.visible;
  }

  dispose(): void {
    window.removeEventListener("keydown", this.keydownHandler);
    this.container.remove();
    const styleEl = document.getElementById(STYLE_ID);
    if (styleEl) styleEl.remove();
  }

  private handleKeydown(e: KeyboardEvent): void {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    if (e.key === "Escape" && this.visible) {
      this.hide();
    } else if ((e.key === "b" || e.key === "B") && this.current) {
      this.toggle();
    }
  }

  private injectStyles(): void {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .intel-briefing .intel-facts li {
        margin: 4px 0;
        padding-left: 14px;
        position: relative;
        color: #d0d0d0;
        font-size: 12px;
        line-height: 1.5;
      }
      .intel-briefing .intel-facts li::before {
        content: "▸";
        position: absolute;
        left: 0;
        color: #00c8ff;
      }
      .intel-briefing .intel-close:hover {
        background: rgba(0, 200, 255, 0.2);
        border-color: rgba(0, 200, 255, 0.7);
        color: #ffffff;
      }
    `;
    document.head.appendChild(style);
  }

  private buildContainer(): HTMLDivElement {
    const panel = document.createElement("div");
    panel.className = "intel-briefing";
    Object.assign(panel.style, {
      position: "absolute",
      top: "16px",
      left: "50%",
      transform: "translateX(-50%)",
      width: "480px",
      maxWidth: "calc(100vw - 32px)",
      background: "rgba(6, 10, 18, 0.92)",
      border: "1px solid rgba(0, 200, 255, 0.4)",
      borderRadius: "6px",
      boxShadow: "0 4px 24px rgba(0, 0, 0, 0.6)",
      backdropFilter: "blur(10px)",
      display: "none",
      flexDirection: "column",
      padding: "16px 20px 14px",
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      color: "#e0e0e0",
      zIndex: "140",
    });

    // Header row with label + close button
    const header = document.createElement("div");
    Object.assign(header.style, {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "4px",
    });

    const tag = document.createElement("div");
    Object.assign(tag.style, {
      color: "#00c8ff",
      fontSize: "10px",
      letterSpacing: "2px",
      textTransform: "uppercase",
      fontWeight: "700",
    });
    tag.textContent = "Intel Briefing";

    const closeBtn = document.createElement("button");
    closeBtn.className = "intel-close";
    closeBtn.textContent = "×";
    Object.assign(closeBtn.style, {
      background: "rgba(0, 200, 255, 0.08)",
      border: "1px solid rgba(0, 200, 255, 0.3)",
      color: "#b0b0b0",
      width: "22px",
      height: "22px",
      borderRadius: "4px",
      fontSize: "16px",
      lineHeight: "1",
      cursor: "pointer",
      fontFamily: "inherit",
      padding: "0",
      transition: "background 0.15s, border-color 0.15s, color 0.15s",
    });
    closeBtn.addEventListener("click", () => this.hide());

    header.appendChild(tag);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    const title = document.createElement("div");
    title.className = "intel-title";
    Object.assign(title.style, {
      fontSize: "17px",
      fontWeight: "700",
      color: "#ffffff",
      margin: "2px 0 8px",
      letterSpacing: "0.2px",
    });
    panel.appendChild(title);

    const description = document.createElement("div");
    description.className = "intel-description";
    Object.assign(description.style, {
      fontSize: "13px",
      color: "#c0c0c0",
      lineHeight: "1.5",
      marginBottom: "10px",
    });
    panel.appendChild(description);

    const factsLabel = document.createElement("div");
    Object.assign(factsLabel.style, {
      fontSize: "10px",
      color: "#888",
      textTransform: "uppercase",
      letterSpacing: "1.5px",
      marginBottom: "4px",
    });
    factsLabel.textContent = "Known Facts";
    panel.appendChild(factsLabel);

    const facts = document.createElement("ul");
    facts.className = "intel-facts";
    Object.assign(facts.style, {
      listStyle: "none",
      padding: "0",
      margin: "0 0 10px",
    });
    panel.appendChild(facts);

    const survival = document.createElement("div");
    survival.className = "intel-survival";
    Object.assign(survival.style, {
      fontSize: "11px",
      color: "#ffaa00",
      fontWeight: "600",
      letterSpacing: "0.5px",
      borderTop: "1px solid rgba(255, 255, 255, 0.08)",
      paddingTop: "8px",
    });
    panel.appendChild(survival);

    return panel;
  }
}
