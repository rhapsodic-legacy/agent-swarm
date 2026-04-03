/**
 * Settings panel for configuring simulation environment parameters.
 * Slides in from the right side of the screen. Toggle with "S" key.
 */

export interface SimSettings {
  terrain_size: number;
  drone_count: number;
  survivor_count: number;
  max_elevation: number;
  drone_speed: number;
  sensor_range: number;
  comms_range: number;
  battery_drain: number;
  day_length: number;
}

interface ParameterDef {
  key: keyof SimSettings;
  label: string;
  min: number;
  max: number;
  default: number;
  step: number;
}

interface Preset {
  name: string;
  values: Partial<SimSettings>;
}

const PARAMETERS: ParameterDef[] = [
  { key: "terrain_size", label: "Terrain Size", min: 64, max: 1024, default: 512, step: 64 },
  { key: "drone_count", label: "Drone Count", min: 4, max: 100, default: 20, step: 1 },
  { key: "survivor_count", label: "Survivors", min: 3, max: 80, default: 25, step: 1 },
  { key: "max_elevation", label: "Max Elevation", min: 50, max: 400, default: 200, step: 10 },
  { key: "drone_speed", label: "Drone Speed (m/s)", min: 5, max: 30, default: 15, step: 1 },
  { key: "sensor_range", label: "Sensor Range (m)", min: 15, max: 80, default: 40, step: 5 },
  { key: "comms_range", label: "Comms Range (m)", min: 40, max: 200, default: 100, step: 10 },
  { key: "battery_drain", label: "Battery Drain (%/s)", min: 0.01, max: 0.2, default: 0.05, step: 0.01 },
  { key: "day_length", label: "Day Length (s)", min: 60, max: 600, default: 300, step: 30 },
];

const PRESETS: Preset[] = [
  { name: "Small", values: { terrain_size: 128, drone_count: 10, survivor_count: 8 } },
  { name: "Medium", values: { terrain_size: 512, drone_count: 20, survivor_count: 25 } },
  { name: "Large", values: { terrain_size: 768, drone_count: 40, survivor_count: 40 } },
  { name: "Massive", values: { terrain_size: 1024, drone_count: 60, survivor_count: 60 } },
];

const STYLE_ID = "settings-panel-styles";

export class SettingsPanel {
  private container: HTMLDivElement;
  private visible = false;
  private settings: SimSettings;
  private sliders: Map<keyof SimSettings, HTMLInputElement> = new Map();
  private valueDisplays: Map<keyof SimSettings, HTMLSpanElement> = new Map();
  private onApply: (config: SimSettings) => void;
  private keydownHandler: (e: KeyboardEvent) => void;

  constructor(onApply: (config: SimSettings) => void) {
    this.onApply = onApply;

    // Initialize settings with defaults
    this.settings = {} as SimSettings;
    for (const param of PARAMETERS) {
      this.settings[param.key] = param.default;
    }

    this.injectStyles();
    this.container = this.buildPanel();
    document.getElementById("app")!.appendChild(this.container);

    // Global keyboard listener
    this.keydownHandler = (e: KeyboardEvent) => this.handleKeydown(e);
    window.addEventListener("keydown", this.keydownHandler);
  }

  toggle(): void {
    this.visible = !this.visible;
    this.container.style.display = this.visible ? "flex" : "none";
  }

  isVisible(): boolean {
    return this.visible;
  }

  dispose(): void {
    window.removeEventListener("keydown", this.keydownHandler);
    this.container.remove();

    const styleEl = document.getElementById(STYLE_ID);
    if (styleEl) {
      styleEl.remove();
    }
  }

  private handleKeydown(e: KeyboardEvent): void {
    // Ignore when typing in an input element
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      return;
    }

    if (e.key === "s" || e.key === "S") {
      this.toggle();
    } else if (e.key === "Escape" && this.visible) {
      this.visible = false;
      this.container.style.display = "none";
    }
  }

  private injectStyles(): void {
    if (document.getElementById(STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .settings-panel input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        width: 100%;
        height: 4px;
        background: #333;
        border-radius: 2px;
        outline: none;
        cursor: pointer;
        margin: 6px 0 0 0;
      }

      .settings-panel input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #00c8ff;
        border: none;
        cursor: pointer;
        box-shadow: 0 0 6px rgba(0, 200, 255, 0.4);
      }

      .settings-panel input[type="range"]::-moz-range-thumb {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #00c8ff;
        border: none;
        cursor: pointer;
        box-shadow: 0 0 6px rgba(0, 200, 255, 0.4);
      }

      .settings-panel input[type="range"]::-webkit-slider-runnable-track {
        height: 4px;
        border-radius: 2px;
      }

      .settings-panel input[type="range"]::-moz-range-track {
        height: 4px;
        background: #333;
        border-radius: 2px;
        border: none;
      }

      .settings-panel input[type="range"]:hover::-webkit-slider-thumb {
        box-shadow: 0 0 10px rgba(0, 200, 255, 0.7);
      }

      .settings-panel input[type="range"]:hover::-moz-range-thumb {
        box-shadow: 0 0 10px rgba(0, 200, 255, 0.7);
      }
    `;
    document.head.appendChild(style);
  }

  private buildPanel(): HTMLDivElement {
    const panel = document.createElement("div");
    panel.className = "settings-panel";
    Object.assign(panel.style, {
      position: "absolute",
      top: "0",
      right: "0",
      bottom: "0",
      width: "300px",
      background: "rgba(0, 0, 0, 0.85)",
      border: "1px solid rgba(0, 200, 255, 0.3)",
      borderRight: "none",
      backdropFilter: "blur(8px)",
      display: "none",
      flexDirection: "column",
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      fontSize: "13px",
      color: "#e0e0e0",
      zIndex: "100",
      overflow: "hidden",
    });

    // Prevent keyboard events on the panel from propagating to the scene
    panel.addEventListener("keydown", (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT") {
        e.stopPropagation();
      }
    });

    // Title
    const title = document.createElement("div");
    Object.assign(title.style, {
      padding: "14px 16px 10px",
      color: "#00c8ff",
      fontSize: "13px",
      letterSpacing: "1.5px",
      textTransform: "uppercase",
      fontWeight: "700",
      borderBottom: "1px solid rgba(0, 200, 255, 0.2)",
      flexShrink: "0",
    });
    title.textContent = "Simulation Settings";
    panel.appendChild(title);

    // Scrollable body
    const body = document.createElement("div");
    Object.assign(body.style, {
      flex: "1",
      overflowY: "auto",
      overflowX: "hidden",
      padding: "12px 16px",
    });

    // Presets
    body.appendChild(this.buildPresetSection());

    // Separator
    body.appendChild(this.buildSeparator());

    // Parameter sliders
    for (const param of PARAMETERS) {
      body.appendChild(this.buildParameterRow(param));
    }

    panel.appendChild(body);

    // Action buttons
    panel.appendChild(this.buildActionBar());

    return panel;
  }

  private buildPresetSection(): HTMLDivElement {
    const section = document.createElement("div");
    Object.assign(section.style, {
      marginBottom: "4px",
    });

    const label = document.createElement("div");
    Object.assign(label.style, {
      fontSize: "11px",
      color: "#888",
      textTransform: "uppercase",
      letterSpacing: "1px",
      marginBottom: "8px",
    });
    label.textContent = "Presets";
    section.appendChild(label);

    const row = document.createElement("div");
    Object.assign(row.style, {
      display: "flex",
      gap: "6px",
      flexWrap: "wrap",
    });

    for (const preset of PRESETS) {
      const btn = document.createElement("button");
      btn.textContent = preset.name;
      Object.assign(btn.style, {
        background: "rgba(0, 200, 255, 0.08)",
        border: "1px solid rgba(0, 200, 255, 0.25)",
        color: "#b0b0b0",
        padding: "5px 12px",
        borderRadius: "4px",
        fontSize: "12px",
        cursor: "pointer",
        transition: "background 0.15s, border-color 0.15s, color 0.15s",
        fontFamily: "inherit",
        lineHeight: "1.3",
      });

      btn.addEventListener("mouseenter", () => {
        btn.style.background = "rgba(0, 200, 255, 0.18)";
        btn.style.borderColor = "rgba(0, 200, 255, 0.5)";
        btn.style.color = "#e0e0e0";
      });
      btn.addEventListener("mouseleave", () => {
        btn.style.background = "rgba(0, 200, 255, 0.08)";
        btn.style.borderColor = "rgba(0, 200, 255, 0.25)";
        btn.style.color = "#b0b0b0";
      });

      btn.addEventListener("click", () => this.applyPreset(preset));
      row.appendChild(btn);
    }

    section.appendChild(row);
    return section;
  }

  private buildSeparator(): HTMLDivElement {
    const sep = document.createElement("div");
    Object.assign(sep.style, {
      height: "1px",
      background: "rgba(0, 200, 255, 0.15)",
      margin: "10px 0",
    });
    return sep;
  }

  private buildParameterRow(param: ParameterDef): HTMLDivElement {
    const row = document.createElement("div");
    Object.assign(row.style, {
      marginBottom: "14px",
    });

    // Label + value header
    const header = document.createElement("div");
    Object.assign(header.style, {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "baseline",
      marginBottom: "0",
    });

    const labelEl = document.createElement("span");
    Object.assign(labelEl.style, {
      color: "#b0b0b0",
      fontSize: "12px",
    });
    labelEl.textContent = param.label;

    const valueEl = document.createElement("span");
    Object.assign(valueEl.style, {
      color: "#00c8ff",
      fontSize: "12px",
      fontWeight: "600",
      fontVariantNumeric: "tabular-nums",
    });
    valueEl.textContent = this.formatValue(param.key, param.default);

    header.appendChild(labelEl);
    header.appendChild(valueEl);
    row.appendChild(header);

    // Slider
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(param.min);
    slider.max = String(param.max);
    slider.step = String(param.step);
    slider.value = String(param.default);

    slider.addEventListener("input", () => {
      const val = parseFloat(slider.value);
      this.settings[param.key] = val;
      valueEl.textContent = this.formatValue(param.key, val);
    });

    row.appendChild(slider);

    this.sliders.set(param.key, slider);
    this.valueDisplays.set(param.key, valueEl);

    return row;
  }

  private buildActionBar(): HTMLDivElement {
    const bar = document.createElement("div");
    Object.assign(bar.style, {
      padding: "12px 16px",
      borderTop: "1px solid rgba(0, 200, 255, 0.2)",
      display: "flex",
      gap: "8px",
      flexShrink: "0",
    });

    // Apply & Reset button
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply & Reset";
    Object.assign(applyBtn.style, {
      flex: "1",
      background: "rgba(0, 200, 255, 0.15)",
      border: "1px solid rgba(0, 200, 255, 0.5)",
      color: "#00c8ff",
      padding: "8px 12px",
      borderRadius: "6px",
      fontSize: "13px",
      fontWeight: "600",
      cursor: "pointer",
      transition: "background 0.15s, border-color 0.15s",
      fontFamily: "inherit",
    });

    applyBtn.addEventListener("mouseenter", () => {
      applyBtn.style.background = "rgba(0, 200, 255, 0.25)";
      applyBtn.style.borderColor = "rgba(0, 200, 255, 0.8)";
    });
    applyBtn.addEventListener("mouseleave", () => {
      applyBtn.style.background = "rgba(0, 200, 255, 0.15)";
      applyBtn.style.borderColor = "rgba(0, 200, 255, 0.5)";
    });

    applyBtn.addEventListener("click", () => {
      this.onApply({ ...this.settings });
    });

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    Object.assign(closeBtn.style, {
      background: "rgba(255, 255, 255, 0.05)",
      border: "1px solid rgba(255, 255, 255, 0.15)",
      color: "#b0b0b0",
      padding: "8px 16px",
      borderRadius: "6px",
      fontSize: "13px",
      cursor: "pointer",
      transition: "background 0.15s, border-color 0.15s, color 0.15s",
      fontFamily: "inherit",
    });

    closeBtn.addEventListener("mouseenter", () => {
      closeBtn.style.background = "rgba(255, 255, 255, 0.1)";
      closeBtn.style.borderColor = "rgba(255, 255, 255, 0.3)";
      closeBtn.style.color = "#e0e0e0";
    });
    closeBtn.addEventListener("mouseleave", () => {
      closeBtn.style.background = "rgba(255, 255, 255, 0.05)";
      closeBtn.style.borderColor = "rgba(255, 255, 255, 0.15)";
      closeBtn.style.color = "#b0b0b0";
    });

    closeBtn.addEventListener("click", () => {
      this.visible = false;
      this.container.style.display = "none";
    });

    bar.appendChild(applyBtn);
    bar.appendChild(closeBtn);
    return bar;
  }

  private applyPreset(preset: Preset): void {
    for (const [key, value] of Object.entries(preset.values) as [keyof SimSettings, number][]) {
      this.settings[key] = value;

      const slider = this.sliders.get(key);
      if (slider) {
        slider.value = String(value);
      }

      const display = this.valueDisplays.get(key);
      if (display) {
        display.textContent = this.formatValue(key, value);
      }
    }
  }

  private formatValue(key: keyof SimSettings, value: number): string {
    if (key === "battery_drain") {
      return value.toFixed(2);
    }
    return String(value);
  }
}
