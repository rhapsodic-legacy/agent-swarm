# Frontend — Three.js 3D Visualization

## Running

```bash
npm run dev       # Start Vite dev server (port 5173)
npm run build     # Production build to dist/
npx vitest run    # Run tests
```

## TypeScript Conventions

- **Strict mode** enabled in tsconfig.json
- **Formatter**: Prettier (configured in .prettierrc)
- **No `any` types** — use `unknown` and narrow
- **Barrel exports**: Each directory has an `index.ts` re-exporting its public API
- **Naming**: PascalCase for classes/types, camelCase for functions/variables, UPPER_SNAKE for constants

## Three.js Patterns

### Performance Rules (Critical)

- **Never create geometries or materials in the render loop.** Create once, reuse.
- **Use InstancedMesh** for drones (50-100 instances from a single draw call).
- **Dispose** geometries, materials, and textures when removing them from scene.
- **Use BufferGeometry** exclusively (never legacy Geometry).
- **Fog-of-war** is a DataTexture updated per frame, applied as a shader overlay — NOT individual mesh modifications.

### Scene Structure

```
Scene
├── Terrain (PlaneGeometry with vertex displacement + vertex colors)
├── Water (PlaneGeometry with transparency, at fixed Y level)
├── DroneInstances (InstancedMesh — all drones in one draw call)
├── SurvivorMarkers (InstancedMesh — discovered survivors)
├── CommLinks (LineSegments — communication lines between drones)
├── SensorCones (InstancedMesh — semi-transparent detection cones)
├── FogOverlay (fullscreen quad with DataTexture)
├── Lights
│   ├── DirectionalLight (sun)
│   ├── AmbientLight (fill)
│   └── HemisphereLight (sky/ground)
└── Sky (gradient background or ShaderMaterial)
```

### Camera Modes

- **Orbit**: OrbitControls, default mode, free rotation/zoom/pan
- **Follow**: Camera tracks a selected drone with smooth lerp
- **Overview**: Top-down orthographic, strategic map view

### Render Pipeline

```
requestAnimationFrame loop:
    1. Receive latest state from WebSocket (or interpolate between last two states)
    2. Update InstancedMesh matrices for drones (position, rotation)
    3. Update fog-of-war DataTexture
    4. Update comm link geometry
    5. Update UI overlays (HUD, minimap)
    6. renderer.render(scene, camera)
```

## WebSocket Client

Located in `src/network/`. Connects to backend at `ws://localhost:8765`.

- Auto-reconnect with exponential backoff
- Parse incoming state snapshots (JSON → typed TypeScript objects)
- Send commands as JSON messages
- See `shared/protocol.md` for message schema

## UI Components

Located in `src/ui/`. HTML/CSS overlays on top of the Three.js canvas.

- **HUD**: Drone count, survivors found, coverage %, elapsed time, selected drone info
- **Minimap**: Top-down 2D canvas showing terrain + drone positions + fog
- **CommandPanel**: Chat input + zone drawing tools + drone selection tools
- **DebugPanel**: Agent decision logs, LLM call history, communication graph
