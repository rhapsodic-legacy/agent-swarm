/**
 * Mission Score panel — tight live readout of the scorecard signals.
 *
 * Coverage % has become the wrong metric for Bayesian search (we want
 * the swarm to *concentrate*, not sweep). This panel surfaces the
 * numbers that actually show whether a run is going well:
 *
 *   • found ratio + survival-window hit rate
 *   • MTTD (mean time to discovery)
 *   • PoC entropy drop (posterior narrowing)
 *   • evidence progress + evidence→survivor latency
 *   • total drone-km flown
 *
 * Positioned below the HUD, above the minimap. Toggle with 'M'.
 */

import type { MetricsInfo } from "@/network/types";

const STYLE_ID = "mission-score-styles";

function fmtTime(t: number | null | undefined): string {
  if (t === null || t === undefined) return "—";
  if (t < 60) return `${t.toFixed(0)}s`;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}m${s.toString().padStart(2, "0")}s`;
}

function fmtPct(p: number | null | undefined): string {
  if (p === null || p === undefined) return "—";
  return `${p.toFixed(1)}%`;
}

function fmtNum(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined) return "—";
  return n.toFixed(digits);
}

export class MissionScore {
  private container: HTMLDivElement;
  private rows: Map<string, HTMLDivElement> = new Map();
  private visible = true;
  private keydownHandler: (e: KeyboardEvent) => void;

  constructor() {
    this.injectStyles();
    this.container = this.build();
    document.getElementById("app")!.appendChild(this.container);

    this.keydownHandler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "m" || e.key === "M") this.toggle();
    };
    window.addEventListener("keydown", this.keydownHandler);
  }

  update(metrics: MetricsInfo | undefined): void {
    if (!metrics) return;
    const l = metrics.latest;
    const q = metrics.search_quality;
    const ev = metrics.evidence_progress;
    const r = metrics.resources;

    this.setRow(
      "found",
      `${l.survivors_found}/${l.total_survivors}`,
      q.survival_window_pct !== null
        ? `${fmtPct(q.survival_window_pct)} in window`
        : "window —",
    );
    this.setRow("mttd", fmtTime(q.mttd_seconds), "mean time to find");
    this.setRow(
      "first",
      fmtTime(metrics.time_to_first_discovery),
      "first survivor",
    );
    this.setRow(
      "evidence",
      `${ev.discovered}/${ev.planted}`,
      metrics.evidence_to_survivor_latency !== null
        ? `→ survivor +${fmtTime(metrics.evidence_to_survivor_latency)}`
        : metrics.time_to_first_evidence !== null
          ? `first ${fmtTime(metrics.time_to_first_evidence)}`
          : "no clues yet",
    );
    this.setRow(
      "entropy",
      q.entropy_drop_pct !== null ? `-${fmtPct(q.entropy_drop_pct)}` : "—",
      "PoC narrowing",
    );
    this.setRow(
      "drones",
      `${fmtNum(r.total_drone_km)} km`,
      r.active_fraction !== null
        ? `${fmtPct(r.active_fraction * 100)} active`
        : "—",
    );
  }

  show(): void {
    this.visible = true;
    this.container.style.display = "block";
  }

  hide(): void {
    this.visible = false;
    this.container.style.display = "none";
  }

  toggle(): void {
    if (this.visible) this.hide();
    else this.show();
  }

  dispose(): void {
    window.removeEventListener("keydown", this.keydownHandler);
    this.container.remove();
    const styleEl = document.getElementById(STYLE_ID);
    if (styleEl) styleEl.remove();
  }

  private setRow(key: string, value: string, detail: string): void {
    const row = this.rows.get(key);
    if (!row) return;
    const vEl = row.querySelector(".score-value") as HTMLSpanElement;
    const dEl = row.querySelector(".score-detail") as HTMLSpanElement;
    vEl.textContent = value;
    dEl.textContent = detail;
  }

  private injectStyles(): void {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .mission-score-panel {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: #e0e0e0;
      }
      .mission-score-panel .score-label {
        color: #888;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-right: 8px;
        min-width: 70px;
        display: inline-block;
      }
      .mission-score-panel .score-value {
        color: #ffffff;
        font-weight: 600;
        font-size: 13px;
        font-variant-numeric: tabular-nums;
        margin-right: 8px;
      }
      .mission-score-panel .score-detail {
        color: #9aa;
        font-size: 11px;
      }
    `;
    document.head.appendChild(style);
  }

  private build(): HTMLDivElement {
    const panel = document.createElement("div");
    panel.className = "mission-score-panel";
    Object.assign(panel.style, {
      position: "absolute",
      top: "16px",
      right: "16px",
      width: "250px",
      background: "rgba(6, 10, 18, 0.88)",
      border: "1px solid rgba(0, 200, 255, 0.3)",
      borderRadius: "6px",
      padding: "10px 12px",
      backdropFilter: "blur(8px)",
      zIndex: "145",
    });

    const tag = document.createElement("div");
    Object.assign(tag.style, {
      color: "#00c8ff",
      fontSize: "10px",
      letterSpacing: "1.5px",
      textTransform: "uppercase",
      fontWeight: "700",
      marginBottom: "6px",
    });
    tag.textContent = "Mission Score";
    panel.appendChild(tag);

    const rows: Array<[string, string]> = [
      ["found", "Found"],
      ["mttd", "MTTD"],
      ["first", "First"],
      ["evidence", "Evidence"],
      ["entropy", "Entropy"],
      ["drones", "Drones"],
    ];
    for (const [key, label] of rows) {
      const row = document.createElement("div");
      Object.assign(row.style, {
        display: "flex",
        alignItems: "baseline",
        padding: "2px 0",
      });
      const lEl = document.createElement("span");
      lEl.className = "score-label";
      lEl.textContent = label;
      const vEl = document.createElement("span");
      vEl.className = "score-value";
      vEl.textContent = "—";
      const dEl = document.createElement("span");
      dEl.className = "score-detail";
      dEl.textContent = "";
      row.appendChild(lEl);
      row.appendChild(vEl);
      row.appendChild(dEl);
      panel.appendChild(row);
      this.rows.set(key, row);
    }

    return panel;
  }
}
