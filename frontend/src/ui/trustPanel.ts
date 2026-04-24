/**
 * TrustPanel — operator autonomy slider (Phase 5D).
 *
 * Sits in the top-right corner next to the zone panel. Adjusts how much
 * the swarm weights operator-placed signals (zones + intel pins) vs its
 * own PoC-based judgment. Also surfaces the adaptive layer's learned
 * multipliers so the operator can see what the system is "thinking."
 *
 *   0.0  → "operator hints are purely advisory"
 *   1.0  → default balance
 *   3.0  → "operator is almost always right"
 */

import type { SwarmClient } from "@/network/client";

export interface AdaptiveSummary {
  operator_trust: number;
  learned_source_scale?: Record<string, number>;
  learned_switching_cost?: Record<string, number>;
  recent_outcome_count?: number;
  recent_outcome_mix?: Record<string, number>;
}

const TRUST_MIN = 0.0;
const TRUST_MAX = 3.0;
const DEFAULT_TRUST = 1.0;

export class TrustPanel {
  private container: HTMLDivElement;
  private slider: HTMLInputElement;
  private valueLabel: HTMLSpanElement;
  private detailsEl: HTMLDivElement;
  private client: SwarmClient;
  private lastSentTrust: number = DEFAULT_TRUST;

  constructor(client: SwarmClient) {
    this.client = client;

    this.container = document.createElement("div");
    Object.assign(this.container.style, {
      position: "absolute",
      top: "320px",
      right: "16px",
      width: "220px",
      padding: "10px 12px",
      background: "rgba(10, 10, 26, 0.88)",
      border: "1px solid rgba(0, 255, 255, 0.25)",
      borderRadius: "6px",
      color: "#e0e0e0",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      fontSize: "11px",
      lineHeight: "1.5",
      pointerEvents: "auto",
      zIndex: "90",
      backdropFilter: "blur(4px)",
      boxShadow: "0 2px 12px rgba(0, 0, 0, 0.4)",
    });

    const heading = document.createElement("div");
    Object.assign(heading.style, {
      fontSize: "11px",
      color: "#00ffff",
      fontWeight: "700",
      letterSpacing: "1px",
      marginBottom: "6px",
      textTransform: "uppercase",
    });
    heading.textContent = "Operator Trust";
    this.container.appendChild(heading);

    const row = document.createElement("div");
    Object.assign(row.style, {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      marginBottom: "8px",
    });

    this.slider = document.createElement("input");
    this.slider.type = "range";
    this.slider.min = String(TRUST_MIN);
    this.slider.max = String(TRUST_MAX);
    this.slider.step = "0.1";
    this.slider.value = String(DEFAULT_TRUST);
    Object.assign(this.slider.style, {
      flex: "1",
      accentColor: "#00ffff",
    });
    this.slider.addEventListener("input", () => this.onSliderInput());
    row.appendChild(this.slider);

    this.valueLabel = document.createElement("span");
    this.valueLabel.textContent = DEFAULT_TRUST.toFixed(1) + "×";
    Object.assign(this.valueLabel.style, {
      minWidth: "36px",
      textAlign: "right",
      color: "#00ffff",
      fontWeight: "600",
    });
    row.appendChild(this.valueLabel);

    this.container.appendChild(row);

    // Explainer line
    const explain = document.createElement("div");
    Object.assign(explain.style, { fontSize: "10px", color: "#888", marginBottom: "6px" });
    explain.textContent = "How much drones weight your hints vs their own PoC map.";
    this.container.appendChild(explain);

    this.detailsEl = document.createElement("div");
    Object.assign(this.detailsEl.style, {
      fontSize: "10px",
      color: "#aaa",
      marginTop: "4px",
      paddingTop: "6px",
      borderTop: "1px solid rgba(255,255,255,0.08)",
    });
    this.detailsEl.textContent = "Awaiting telemetry…";
    this.container.appendChild(this.detailsEl);

    document.body.appendChild(this.container);
  }

  /** Called from main render loop with the latest adaptive summary. */
  update(summary: AdaptiveSummary | undefined): void {
    if (!summary) return;
    // Snap slider to the server's current trust if the operator isn't
    // actively dragging (avoids fighting the user mid-drag).
    if (document.activeElement !== this.slider) {
      const incoming = Number(summary.operator_trust ?? DEFAULT_TRUST);
      if (Math.abs(incoming - Number(this.slider.value)) > 0.01) {
        this.slider.value = String(incoming);
        this.valueLabel.textContent = incoming.toFixed(1) + "×";
      }
    }

    const learned = summary.learned_source_scale ?? {};
    const entries = Object.entries(learned);
    if (entries.length === 0) {
      this.detailsEl.innerHTML = `<span style="color:#666">No learning yet.</span>`;
    } else {
      const rows = entries
        .map(([k, v]) => {
          const color = v > 1.05 ? "#44ff88" : v < 0.95 ? "#ff8844" : "#aaa";
          return `<div style="display:flex;justify-content:space-between;color:${color}"><span>${k}</span><span>${v.toFixed(2)}×</span></div>`;
        })
        .join("");
      this.detailsEl.innerHTML =
        `<div style="color:#888;margin-bottom:3px">Learned adjustments:</div>${rows}`;
    }
  }

  private onSliderInput(): void {
    const v = Number(this.slider.value);
    this.valueLabel.textContent = v.toFixed(1) + "×";
    // Debounce via quick threshold: only emit when it actually moved a bit
    if (Math.abs(v - this.lastSentTrust) < 0.05) return;
    this.lastSentTrust = v;
    this.client.sendTrustCommand(v);
  }
}
