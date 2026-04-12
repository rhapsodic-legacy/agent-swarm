/**
 * Help overlay — shows all keybindings and controls.
 * Toggle with ? or F1 key.
 */

export class HelpOverlay {
  private container: HTMLDivElement;
  private visible = false;

  constructor() {
    this.container = document.createElement("div");
    Object.assign(this.container.style, {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      background: "rgba(0, 0, 0, 0.9)",
      border: "1px solid rgba(0, 200, 255, 0.4)",
      borderRadius: "12px",
      padding: "24px 32px",
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      fontSize: "13px",
      color: "#e0e0e0",
      display: "none",
      zIndex: "300",
      backdropFilter: "blur(12px)",
      maxWidth: "480px",
      lineHeight: "1.6",
    });

    this.container.innerHTML = `
      <h2 style="color:#00c8ff;font-size:16px;margin:0 0 16px 0;letter-spacing:1px;text-transform:uppercase">Controls</h2>
      <table style="border-collapse:collapse;width:100%">
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">WASD</td><td>Pan camera</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">Right-click drag</td><td>Pan camera</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">Left-click drag</td><td>Rotate camera</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">Scroll</td><td>Zoom in/out</td></tr>
        <tr><td colspan="2" style="padding:8px 0 4px 0;border-top:1px solid rgba(255,255,255,0.1)"></td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">Space</td><td>Pause / Resume</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">1 / 2 / 3</td><td>Speed 1x / 2x / 5x</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">N</td><td>New simulation (reset)</td></tr>
        <tr><td colspan="2" style="padding:8px 0 4px 0;border-top:1px solid rgba(255,255,255,0.1)"></td></tr>
        <tr><td style="color:#ff4400;padding:3px 16px 3px 0;font-weight:600">G</td><td>God Mode — reveal all survivor positions</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">L</td><td>Toggle activity log</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">T</td><td>Toggle chat panel</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">S</td><td>Toggle settings panel</td></tr>
        <tr><td colspan="2" style="padding:8px 0 4px 0;border-top:1px solid rgba(255,255,255,0.1)"></td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">Left-click</td><td>Select drone</td></tr>
        <tr><td style="color:#00c8ff;padding:3px 16px 3px 0;font-weight:600">Right-click</td><td>Send selected drone to location</td></tr>
      </table>
      <div style="margin-top:16px;color:#666;font-size:11px;text-align:center">Press <span style="color:#00c8ff">?</span> or <span style="color:#00c8ff">F1</span> to close</div>
    `;

    document.getElementById("app")!.appendChild(this.container);
  }

  toggle(): void {
    this.visible = !this.visible;
    this.container.style.display = this.visible ? "block" : "none";
  }

  isVisible(): boolean {
    return this.visible;
  }

  dispose(): void {
    this.container.remove();
  }
}
