# Gargantua — Real-Time Kerr Black Hole Simulator (WebGPU)

Real-time relativistic black hole simulator built in Rust (`wgpu` + `winit` + `egui`) compiled to WebAssembly + WebGPU.

## Features
- **Horizon-penetrating Cartesian Kerr-Schild geodesics** with curvature-adaptive RK4 integration and exact weak-field asymptotic escape tail
- **Pure-GR nested photon rings** ($n = 1, 2, 3\dots$) resolved cleanly via a dedicated 5× rotated-grid sub-pixel critical-curve pass (`photon_ssaa.comp.wgsl`) with zero synthetic glow
- **Liouville-exact gravitational starfield lensing** across a 22,000-star catalog + procedural micro-stars with central-difference 2×2 EWA Jacobians (preserving Einstein-ring surface brightness without bitmap pixelation)
- **Gargantua-inspired volumetric accretion disk** with sheared multi-scale filaments, self-occluding dust lanes, relativistic Doppler beaming & gravitational redshift, plus a crescent-lit orbiting planet
- **6-mip Jimenez HDR bloom & warm AgX tone mapping** over a true obsidian-black vacuum background

## Controls
- `W` / `S` / `A` / `D` / `Q` / `E`: Move forward / back / left / right / up / down (distance-adaptive speed slows smoothly near the horizon & singularity)
- **Left-click & drag** (or `L` / `Esc` for pointer lock): Look around
- `Shift + W/S/A/D/Q/E` or **Arrow keys**: Pitch, yaw, and roll without translating
- `Ctrl` / `Space`: 0.25× fine control / 3× boost
- `R`: Reset to hero Gargantua framing
- `H`: Toggle HUD overlay
- `P` / `F12`: Download a lossless PNG snapshot

## Browser Requirements
Requires a WebGPU-enabled browser (desktop Chrome, Edge, Brave, Arc, or Safari 18+ with hardware acceleration enabled).
