#!/usr/bin/env node
/**
 * Visual verification via independent sub-agents.
 *
 * Each agent connects to the live app, reads raw pixel data, and reports
 * what it observes — sizes, colors, positions — without knowing what the
 * "correct" answer is. The aggregated reports form ground truth.
 *
 * Usage:
 *   1. Start backend + frontend
 *   2. node scripts/visual_agent_verify.mjs
 *
 * Agents:
 *   1. terrain_agent    — reports terrain coverage and colors
 *   2. drone_agent      — finds drone-colored pixels, reports count and size
 *   3. survivor_agent   — finds red marker pixels, reports count, size, positions
 *   4. scale_agent      — compares drone pixel size vs survivor pixel size
 */

import puppeteer from "puppeteer";
import { PNG } from "pngjs";

const FRONTEND_URL = "http://localhost:5173";

async function launchPage() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });
  await page.goto(FRONTEND_URL, { waitUntil: "networkidle0", timeout: 30000 });
  await page.waitForFunction(
    () => document.getElementById("connection-status")?.classList.contains("connected"),
    { timeout: 15000 },
  );
  await new Promise((r) => setTimeout(r, 5000));
  return { browser, page };
}

async function readPixels(page) {
  return page.evaluate(async () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    // Wait for two rAF cycles to ensure the latest frame has rendered
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return null;
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return { pixels: Array.from(pixels), w, h };
  });
}

function findBlobs(pixelData, matchFn) {
  const { pixels, w, h } = pixelData;
  // Find connected regions of matching pixels
  const visited = new Set();
  const blobs = [];

  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const idx = (y * w + x) * 4;
      const key = `${x},${y}`;
      if (visited.has(key)) continue;

      const r = pixels[idx], g = pixels[idx + 1], b = pixels[idx + 2];
      if (!matchFn(r, g, b)) continue;

      // Flood fill to find blob extent
      const queue = [[x, y]];
      let minX = x, maxX = x, minY = y, maxY = y;
      let count = 0;
      let totalR = 0, totalG = 0, totalB = 0;

      while (queue.length > 0 && count < 5000) {
        const [cx, cy] = queue.pop();
        const ck = `${cx},${cy}`;
        if (visited.has(ck)) continue;
        if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;

        const ci = (cy * w + cx) * 4;
        const cr = pixels[ci], cg = pixels[ci + 1], cb = pixels[ci + 2];
        if (!matchFn(cr, cg, cb)) continue;

        visited.add(ck);
        count++;
        totalR += cr; totalG += cg; totalB += cb;
        minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);

        // Check neighbors (skip by 2 for speed)
        for (const [dx, dy] of [[2, 0], [-2, 0], [0, 2], [0, -2]]) {
          queue.push([cx + dx, cy + dy]);
        }
      }

      if (count >= 3) {
        blobs.push({
          x: minX, y: minY,
          width: maxX - minX, height: maxY - minY,
          pixelCount: count,
          avgColor: {
            r: Math.round(totalR / count),
            g: Math.round(totalG / count),
            b: Math.round(totalB / count),
          },
          centerX: Math.round((minX + maxX) / 2),
          centerY: Math.round((minY + maxY) / 2),
        });
      }
    }
  }
  return blobs;
}

// ========== AGENT 1: Terrain ==========
async function terrainAgent(page) {
  const data = await readPixels(page);
  if (!data) return { agent: "terrain", error: "no pixel data" };

  const { pixels, w, h } = data;
  let terrainPixels = 0, skyPixels = 0, totalPixels = w * h;
  const colorBuckets = {};

  for (let i = 0; i < pixels.length; i += 16) { // sample every 4th pixel
    const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    // Sky/background is dark gray (~10,10,26)
    if (r < 20 && g < 20 && b < 35) { skyPixels++; continue; }
    terrainPixels++;
    const bucket = `${Math.floor(r / 50) * 50},${Math.floor(g / 50) * 50},${Math.floor(b / 50) * 50}`;
    colorBuckets[bucket] = (colorBuckets[bucket] || 0) + 1;
  }

  const topColors = Object.entries(colorBuckets).sort((a, b) => b[1] - a[1]).slice(0, 5);

  return {
    agent: "terrain",
    canvasSize: `${w}x${h}`,
    terrainCoverage: `${((terrainPixels / (totalPixels / 4)) * 100).toFixed(1)}%`,
    skyCoverage: `${((skyPixels / (totalPixels / 4)) * 100).toFixed(1)}%`,
    topColors: topColors.map(([c, n]) => `RGB(${c}): ${n} samples`),
  };
}

// ========== AGENT 2: Drones ==========
async function droneAgent(page) {
  const data = await readPixels(page);
  if (!data) return { agent: "drones", error: "no pixel data" };

  // Drones are cyan/blue: high B, moderate G, low R
  const isDrone = (r, g, b) => b > 150 && g > 100 && r < 100;
  const blobs = findBlobs(data, isDrone);

  return {
    agent: "drones",
    blobsFound: blobs.length,
    blobs: blobs.slice(0, 10).map((b) => ({
      position: `(${b.centerX}, ${b.centerY})`,
      size: `${b.width}x${b.height}px`,
      pixels: b.pixelCount,
      color: `RGB(${b.avgColor.r}, ${b.avgColor.g}, ${b.avgColor.b})`,
    })),
    avgBlobWidth: blobs.length > 0
      ? Math.round(blobs.reduce((s, b) => s + b.width, 0) / blobs.length)
      : 0,
    avgBlobHeight: blobs.length > 0
      ? Math.round(blobs.reduce((s, b) => s + b.height, 0) / blobs.length)
      : 0,
  };
}

// ========== AGENT 3: Survivors (god mode) ==========
async function survivorAgent(page) {
  // Enable god mode
  await page.keyboard.press("g");
  await new Promise((r) => setTimeout(r, 3000));

  const data = await readPixels(page);
  if (!data) return { agent: "survivors", error: "no pixel data" };

  // Survivors are red: R > 150, G < 80, B < 60
  const isSurvivor = (r, g, b) => r > 150 && g < 80 && b < 60;
  const blobs = findBlobs(data, isSurvivor);

  // Check console for god mode status
  const godModeInfo = await page.evaluate(() => {
    const debug = window.__DEBUG;
    if (!debug) return { error: "no debug hooks" };
    const state = debug.getState?.();
    return {
      godModeActive: debug.godMode?.isActive?.() ?? "unknown",
      allSurvivorsCount: state?.all_survivors?.length ?? 0,
      meshCount: debug.godMode?.mesh?.count ?? "unknown",
    };
  });

  return {
    agent: "survivors",
    godMode: godModeInfo,
    redBlobsFound: blobs.length,
    blobs: blobs.slice(0, 10).map((b) => ({
      position: `(${b.centerX}, ${b.centerY})`,
      size: `${b.width}x${b.height}px`,
      pixels: b.pixelCount,
      color: `RGB(${b.avgColor.r}, ${b.avgColor.g}, ${b.avgColor.b})`,
    })),
    avgBlobWidth: blobs.length > 0
      ? Math.round(blobs.reduce((s, b) => s + b.width, 0) / blobs.length)
      : 0,
    avgBlobHeight: blobs.length > 0
      ? Math.round(blobs.reduce((s, b) => s + b.height, 0) / blobs.length)
      : 0,
    totalRedPixels: blobs.reduce((s, b) => s + b.pixelCount, 0),
  };
}

// ========== AGENT 5: PoC Heatmap (before/after comparison) ==========
// Ground-truth test: measure saturated-color pixel count before and after
// the heatmap toggle. A real heatmap must add a clear quantity of saturated
// (non-terrain, non-sky) pixels.
function countSaturatedPixelsFromPng(pngBuffer) {
  const png = PNG.sync.read(pngBuffer);
  const { width, height, data } = png;
  let count = 0;
  // Sample every 4th pixel for speed (width * height * 4 bytes per pixel)
  for (let i = 0; i < data.length; i += 16) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    if (max > 120 && (max - min) > 60) count++;
  }
  return count;
}

function countSaturatedPixels(pixelData) {
  const { pixels } = pixelData;
  let count = 0;
  for (let i = 0; i < pixels.length; i += 16) {
    const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    // Saturated = at least one channel is high AND there's meaningful
    // separation between the max and min channel (i.e. not gray)
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    if (max > 120 && (max - min) > 60) count++;
  }
  return count;
}

async function heatmapAgent(page, baselineCount) {
  const data = await readPixels(page);
  if (!data) return { agent: "poc_heatmap", error: "no pixel data" };

  const withHeatmap = countSaturatedPixels(data);
  const delta = withHeatmap - baselineCount;
  // A visible heatmap adds many saturated pixels across the terrain plane.
  // Threshold: 1000+ new saturated pixels at 1280x720 sampling every 4th.
  const rendering = delta > 1000;

  return {
    agent: "poc_heatmap",
    baselineSaturated: baselineCount,
    withHeatmapSaturated: withHeatmap,
    delta,
    verdict: rendering
      ? `OK: heatmap adds ${delta} saturated pixels`
      : `FAIL: heatmap added only ${delta} saturated pixels`,
  };
}

// ========== AGENT 4: Scale comparison ==========
async function scaleAgent(droneReport, survivorReport) {
  const droneAvgH = droneReport.avgBlobHeight || 0;
  const survAvgH = survivorReport.avgBlobHeight || 0;
  const droneAvgW = droneReport.avgBlobWidth || 0;
  const survAvgW = survivorReport.avgBlobWidth || 0;

  const heightRatio = droneAvgH > 0 ? (survAvgH / droneAvgH).toFixed(1) : "N/A";
  const widthRatio = droneAvgW > 0 ? (survAvgW / droneAvgW).toFixed(1) : "N/A";

  return {
    agent: "scale_comparison",
    droneAvgSize: `${droneAvgW}x${droneAvgH}px`,
    survivorAvgSize: `${survAvgW}x${survAvgH}px`,
    heightRatio: `${heightRatio}x (target: 3-4x)`,
    widthRatio: `${widthRatio}x`,
    verdict:
      survAvgH === 0
        ? "FAIL: No survivor blobs found"
        : parseFloat(heightRatio) > 6
          ? `FAIL: Survivors too large (${heightRatio}x drone height)`
          : parseFloat(heightRatio) < 1
            ? `FAIL: Survivors too small (${heightRatio}x drone height)`
            : `OK: Survivor/drone ratio = ${heightRatio}x height`,
  };
}

// ========== MAIN ==========
async function main() {
  // Check servers
  try { await fetch("http://localhost:8765/health"); } catch {
    console.error("Backend not running"); process.exit(1);
  }

  console.log("========================================");
  console.log("VISUAL VERIFICATION — INDEPENDENT AGENTS");
  console.log("========================================\n");

  const { browser, page } = await launchPage();

  try {
    // Run agents sequentially (they share the page)
    console.log("--- AGENT 1: Terrain ---");
    const terrainReport = await terrainAgent(page);
    console.log(JSON.stringify(terrainReport, null, 2));

    console.log("\n--- AGENT 2: Drones ---");
    const droneReport = await droneAgent(page);
    console.log(JSON.stringify(droneReport, null, 2));

    console.log("\n--- AGENT 3: Survivors (enabling god mode) ---");
    const survivorReport = await survivorAgent(page);
    console.log(JSON.stringify(survivorReport, null, 2));

    console.log("\n--- AGENT 4: Scale Comparison ---");
    const scaleReport = await scaleAgent(droneReport, survivorReport);
    console.log(JSON.stringify(scaleReport, null, 2));

    console.log("\n--- AGENT 5: PoC Heatmap (Bayesian search overlay) ---");
    // Press G to turn off god mode first (cleaner reading)
    await page.keyboard.press("g");
    await new Promise((r) => setTimeout(r, 1500));
    // Measure baseline via screenshot (ground truth — pixel data matches what user sees)
    const baselinePng = await page.screenshot({ type: "png" });
    const baselineSaturated = countSaturatedPixelsFromPng(baselinePng);
    // Press H to enable heatmap
    await page.keyboard.press("h");
    await new Promise((r) => setTimeout(r, 3000));
    const heatmapPng = await page.screenshot({ type: "png", path: "/tmp/visual_agent_heatmap.png" });
    const withHeatmapSaturated = countSaturatedPixelsFromPng(heatmapPng);
    const heatmapReport = {
      agent: "poc_heatmap",
      baselineSaturated,
      withHeatmapSaturated,
      delta: withHeatmapSaturated - baselineSaturated,
      verdict:
        withHeatmapSaturated - baselineSaturated > 2000
          ? `OK: heatmap adds ${withHeatmapSaturated - baselineSaturated} saturated pixels`
          : `FAIL: heatmap added only ${withHeatmapSaturated - baselineSaturated} saturated pixels`,
    };
    console.log(JSON.stringify(heatmapReport, null, 2));

    // Save screenshot for human inspection (god off, heatmap on)
    await page.screenshot({ path: "/tmp/visual_agent_heatmap.png" });
    console.log("\nHeatmap screenshot: /tmp/visual_agent_heatmap.png");

    // Restore: heatmap off, god on for the main screenshot
    await page.keyboard.press("h");
    await page.keyboard.press("g");
    await new Promise((r) => setTimeout(r, 1000));
    await page.screenshot({ path: "/tmp/visual_agent_verify.png" });
    console.log("Main screenshot: /tmp/visual_agent_verify.png");

    // Final verdict
    console.log("\n========================================");
    console.log("FINAL DOSSIER");
    console.log("========================================");
    console.log(`Terrain: ${terrainReport.terrainCoverage} coverage`);
    console.log(`Drones: ${droneReport.blobsFound} found, avg ${droneReport.avgBlobWidth}x${droneReport.avgBlobHeight}px`);
    console.log(`Survivors: ${survivorReport.redBlobsFound} red blobs, avg ${survivorReport.avgBlobWidth}x${survivorReport.avgBlobHeight}px`);
    console.log(`Scale: ${scaleReport.verdict}`);
    console.log(`God mode data: ${survivorReport.godMode?.allSurvivorsCount ?? 0} survivors in state`);

    const pass =
      survivorReport.redBlobsFound > 0 &&
      scaleReport.verdict.startsWith("OK");
    console.log(`\nVERDICT: ${pass ? "PASS" : "FAIL"}`);
    process.exit(pass ? 0 : 1);
  } finally {
    await browser.close();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
