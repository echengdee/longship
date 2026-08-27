# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the GitHub Pages site for **Perceptive Humanoid Parkour (PHP)** — a research project demonstrating depth-based humanoid robot parkour using motion matching and reinforcement learning. The site combines a static academic paper page with an interactive browser-based MuJoCo physics simulator running ONNX neural network policies.

## Repository Structure

- **Root**: Static GitHub Pages site (`index.html`, `demo.html` + mobile variants) using Bulma CSS
- **`mujoco_wasm/`**: Vite-based web app — the core interactive simulator
- **`static/`**: CSS, JS, images, and videos for the landing page

The root HTML pages embed the simulator via iframes pointing to `mujoco_wasm/dist-desktop/` or `mujoco_wasm/dist-mobile/`. Mobile pages (`index-mobile.html`, `demo-mobile.html`) are auto-redirected to from desktop pages when the viewport is narrow.

## Build Commands

All commands run from `mujoco_wasm/`:

```bash
npm install              # Install dependencies
npm run dev              # Vite dev server (--host enabled)
npm run build:web        # Production build → dist/
npm run build:web:desktop  # Desktop build → dist-desktop/ (VITE_DEPTH_PREVIEW_SCALE=4, VITE_GUI_MODE=open)
npm run build:web:mobile   # Mobile build → dist-mobile/ (VITE_DEPTH_PREVIEW_SCALE=1, VITE_GUI_MODE=hide)
npm run build            # esbuild Node bundle → build/ (used by GitHub Actions)
```

There are no tests or linting configured.

## Environment Variables (Vite)

- `VITE_DEPTH_PREVIEW_SCALE` — Depth camera inset size multiplier (1 for mobile, 4 for desktop)
- `VITE_GUI_MODE` — `"open"` shows lil-gui panel, `"hide"` shows mobile button UI instead

## Architecture (mujoco_wasm/src/)

### main.js — `MuJoCoDemo` class
The application entry point and orchestrator. Manages:
- Three.js renderer, scene graph, cameras (main view + depth sensor)
- MuJoCo WASM integration: loads XML scene models into Emscripten virtual filesystem
- Render loop: steps physics (500Hz), runs policy, syncs body transforms to Three.js meshes
- Camera tracking: follows robot pelvis with configurable offset
- Keyboard input: W/A/D (movement), Y (speed toggle), SPACE (pause), BACKSPACE (reset)

### mujocoUtils.js
Scene loading and GUI utilities:
- `loadSceneFromURL()` — downloads XML, creates MuJoCo model/data, builds Three.js scene graph
- `setupGUI()` — creates lil-gui control panel with simulation parameters
- Coordinate transforms between MuJoCo and Three.js (`toMujocoPos`, `getPosition`, `getQuaternion`)

### policy/policyController.js — `PolicyController` class
ONNX Runtime inference for autonomous robot control:
- Loads two ONNX models: main policy (~12.7 MB) and depth backbone (~105 KB)
- Composes observations from joint state + depth camera features
- Runs at 50Hz (policy decimation over 500Hz physics)
- Applies actions via PD position/velocity controllers
- Manages depth latency queue (7-step buffer) for realistic perception delay
- Parses model metadata for feature name mappings and default values
- Two speed modes toggled by Y key

### utils/
- `DragStateManager.js` — Mouse raycasting and force application to physics bodies
- `Debug.js` — Console error overlay for mobile debugging
- `Reflector.js` — Three.js mirror reflection shader

## Key Data Flow

```
Keyboard input → PolicyController (command vector)
                        ↓
Depth camera render → resize/crop → depth backbone ONNX → depth features
                        ↓
Joint state + depth features → main policy ONNX → target positions
                        ↓
PD controller → MuJoCo ctrl → mj_step() → updated body positions
                        ↓
Three.js scene sync → WebGL render
```

## Assets

- `mujoco_wasm/assets/scenes/` — MuJoCo XML models and mesh files (OBJ, STL). Primary scene: `g1_with_terrain.xml`
- `mujoco_wasm/public/` — Pre-built ONNX model files (copied to dist roots during build)
- `static/videos/` — Demo videos embedded in the landing page

## Deployment

GitHub Actions (`.github/workflows/main.yml`) deploys the entire repo as a static site on push to `main`. The `dist-desktop/` and `dist-mobile/` directories are checked into git and served directly — they are build artifacts that must be rebuilt and committed when source changes.
