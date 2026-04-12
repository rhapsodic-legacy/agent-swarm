#!/usr/bin/env node
/**
 * Visual integration test — launches real browser against real servers,
 * enables god mode, reads canvas pixels, checks for survivor markers.
 *
 * Usage: node scripts/test_visual_survivors.mjs
 * Requires: both backend and frontend servers running
 */

import puppeteer from "puppeteer";

const FRONTEND_URL = "http://localhost:5173";
const HEALTH_URL = "http://localhost:8765/health";

// Survivor marker: MeshBasicMaterial with 0xFF2200 (unaffected by tone mapping).
// Should appear as bright red: R>150, G<80, B<60
function isSurvivorColor(r, g, b) {
  return r > 150 && g < 80 && b < 60;
}

async function main() {
  // Check servers
  try {
    const resp = await fetch(HEALTH_URL);
    if (!resp.ok) throw new Error("not ok");
  } catch {
    console.error("ERROR: Backend not running at", HEALTH_URL);
    process.exit(1);
  }

  try {
    const resp = await fetch(FRONTEND_URL);
    if (!resp.ok) throw new Error("not ok");
  } catch {
    console.error("ERROR: Frontend not running at", FRONTEND_URL);
    process.exit(1);
  }

  console.log("Both servers running. Launching browser...");

  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });

  // Collect console logs
  const logs = [];
  page.on("console", (msg) => {
    const text = msg.text();
    logs.push(text);
    if (text.includes("all_survivors") || text.includes("GodMode") || text.includes("Survivor")) {
      console.log("  [BROWSER]", text);
    }
  });

  console.log("Navigating to", FRONTEND_URL);
  await page.goto(FRONTEND_URL, { waitUntil: "networkidle0", timeout: 30000 });

  // Wait for connection
  await page.waitForFunction(
    () => document.getElementById("connection-status")?.classList.contains("connected"),
    { timeout: 15000 },
  );
  console.log("Connected to backend");

  // Wait for chunks to load
  await new Promise((r) => setTimeout(r, 5000));
  console.log("Waited 5s for chunks");

  // Press G for god mode
  await page.keyboard.press("g");
  console.log("Pressed G — god mode toggled");
  await new Promise((r) => setTimeout(r, 2000));

  // DON'T move camera — test from the default drone fleet view
  // to verify survivors are visible from where the user actually looks
  const moveResult = await page.evaluate(() => {
    const debug = window.__DEBUG;
    if (!debug) return { error: "no __DEBUG on window" };

    const state = debug.getState();
    if (!state) return { error: "no state yet" };

    const allS = state.all_survivors;
    if (!allS || allS.length === 0) return { error: "no all_survivors in state", keys: Object.keys(state) };

    // Move camera to look at the first survivor
    const s = allS[0];
    const sx = s.position[0];
    const sy = s.position[1];
    const sz = s.position[2];

    debug.controls.target.set(sx, sy, sz);
    debug.camera.position.set(sx + 200, sy + 300, sz + 200);
    debug.camera.lookAt(sx, sy, sz);
    debug.controls.update();

    return {
      survivorCount: allS.length,
      cameraTarget: [sx, sy, sz],
      firstSurvivor: s,
      godMeshCount: debug.godMode.mesh?.count ?? "unknown",
    };
  });
  console.log("Camera move result:", JSON.stringify(moveResult, null, 2));

  // Wait for a few frames to render
  await new Promise((r) => setTimeout(r, 3000));

  // Screenshot for human inspection
  await page.screenshot({ path: "/tmp/drone_swarm_godmode.png" });
  console.log("Screenshot saved to /tmp/drone_swarm_godmode.png");

  // Read canvas pixels
  const result = await page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return { error: "no canvas" };

    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return { error: "no webgl context" };

    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Check if all zeros (preserveDrawingBuffer issue)
    let nonZero = 0;
    for (let i = 0; i < Math.min(pixels.length, 10000); i++) {
      if (pixels[i] !== 0) nonZero++;
    }

    // Scan ALL pixels for survivor red
    let survivorPixels = 0;
    let totalNonBlack = 0;
    const redSamples = [];

    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2], a = pixels[i + 3];
      if (r > 10 || g > 10 || b > 10) totalNonBlack++;
      if (r > 150 && g < 80 && b < 60) {
        survivorPixels++;
        if (redSamples.length < 10) {
          const px = (i / 4) % w;
          const py = Math.floor((i / 4) / w);
          redSamples.push({ x: px, y: py, r, g, b });
        }
      }
    }

    return { w, h, nonZero, totalNonBlack, survivorPixels, redSamples };
  });

  console.log("\n=== PIXEL ANALYSIS ===");
  console.log("Canvas:", result.w, "x", result.h);
  console.log("Non-zero pixels (first 10k):", result.nonZero);
  console.log("Non-black pixels:", result.totalNonBlack);
  console.log("SURVIVOR PIXELS (R>180,G<120,B<80):", result.survivorPixels);

  if (result.redSamples && result.redSamples.length > 0) {
    console.log("Sample survivor pixel locations:");
    for (const s of result.redSamples) {
      console.log(`  (${s.x}, ${s.y}): RGB(${s.r}, ${s.g}, ${s.b})`);
    }
  }

  // Print relevant console logs
  const relevantLogs = logs.filter(
    (l) => l.includes("all_survivors") || l.includes("GodMode") || l.includes("[DroneSwarm]"),
  );
  if (relevantLogs.length > 0) {
    console.log("\n=== RELEVANT BROWSER CONSOLE LOGS ===");
    for (const l of relevantLogs.slice(0, 20)) {
      console.log(" ", l);
    }
  }

  // Verdict
  console.log("\n=== VERDICT ===");
  if (result.nonZero === 0) {
    console.log("FAIL: Canvas is all zeros — preserveDrawingBuffer may not be working");
  } else if (result.survivorPixels > 0) {
    console.log(`PASS: ${result.survivorPixels} survivor pixels found on screen`);
  } else {
    console.log("FAIL: No survivor pixels found. Survivors are NOT rendering.");
    console.log("Check: is all_survivors data reaching the GodMode renderer?");
  }

  await browser.close();
  process.exit(result.survivorPixels > 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
