# MLDetect

A website that tracks every moving object in a video (cars on a highway, planes in the sky, boats, birds, cyclists…) and shows, for each one:

- a **bounding box** from the moment it appears, with a stable **ID** for as long as it stays in view;
- its estimated **speed in km/h**;
- its **direction of travel** (on-screen compass heading, plus *approaching* / *receding* when it moves towards or away from the camera);
- its **predicted path** for the next few seconds, plus a trail of where it has been.

Everything runs **in your browser**. The video is never uploaded anywhere.

## Quick start

```bash
npm install
npm run dev        # downloads the models on first run, then serves http://localhost:5173
```

Drop a video onto the page and press play.

| Command | What it does |
| --- | --- |
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the production build |
| `npm test` | Unit tests (Vitest) |
| `npm run fetch-model` | Download the YOLOX models into `public/models/` |

## Using it

- **Live mode** (default): press play and frames are analysed as fast as your device allows. With WebGPU (recent Chrome/Edge) that is usually real time. On the CPU/WASM fallback, lower the playback speed for smoother tracking.
- **Precise analysis**: steps through the whole video at a fixed rate (5–30 fps), analysing every step. You can then replay and scrub with perfectly synced results, and **Export CSV** (one row per object per frame: position, velocity, speed, heading).
- **Model**: *Fast* (YOLOX-Tiny, 416 px) or *Accurate* (YOLOX-S, 640 px, about 3× slower), which finds much smaller and more distant objects.
- **Objects**: road vehicles, aircraft & birds, anything that moves, or all 80 COCO classes.
- Click a box or a table row to highlight one object.

## How it works

```
<video> frame ─► Web Worker: letterbox → YOLOX (ONNX Runtime Web, WebGPU → WASM) → decode + NMS
             ─► Tracker: Kalman filter per object + ByteTrack two-stage association (Hungarian)
             ─► Motion: least-squares velocity, turn rate, pinhole-camera speed estimate
             ─► Canvas overlay (boxes, trails, arrows, predicted paths) + live table
```

- **Detection**: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) COCO models (Apache-2.0), run with ONNX Runtime Web in a Web Worker. It uses WebGPU when available and falls back to multi-threaded WASM.
- **Tracking** (`src/tracking/tracker.ts`): constant-velocity Kalman filter per box. ByteTrack-style matching first pairs confident detections with predicted boxes (IoU, plus a motion-aware distance gate for fast objects at low frame rates), then uses low-confidence detections to keep existing tracks alive. Tracks survive about 1 s of occlusion and are dropped once they leave the frame.
- **Speed** (`src/tracking/motion.ts`): the scale (metres per pixel) comes from each class's typical size (car 4.5 × 1.7 m, bus 12 m, airliner 38 m, …), applied along the object's direction of travel. Motion towards or away from the camera comes from how fast the box grows or shrinks ("looming"), using a pinhole camera model with an assumed ~75° diagonal field of view. Combining the two gives a 3-D speed, so cars driving towards an overpass camera aren't underestimated.
- **Prediction**: constant speed + constant turn rate (CTRV) extrapolation of the recent motion.

### Accuracy and limitations

Speeds are **estimates** (typically within ±20–30% for a static camera):

- **The camera must be static.** If the camera pans or zooms to follow an object, the speed is measured relative to the moving camera.
- Strong zoom (narrow field of view) or unusually sized vehicles (e.g. a semi-truck vs a van, both COCO "truck") bias the scale.
- Very small or distant objects may not be detected; use the *Accurate* model.
- Brand-new tracks show *measuring…* until there is enough motion history.

## Deploying

`.github/workflows/deploy.yml` builds and publishes the site to **GitHub Pages** on every push to `main`. Enable it once under *Settings → Pages → Source: GitHub Actions*. The site is fully static, so any static host works: serve `dist/`, ideally with these headers to enable multi-threaded WASM:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them (e.g. on GitHub Pages) it still works: WebGPU is unaffected and the CPU fallback runs single-threaded.

## Project layout

```
src/detection/   model registry, worker, pre/post-processing (decode, NMS), class sizes
src/tracking/    Kalman filter, Hungarian assignment, ByteTrack tracker, motion & speed
src/engine/      TrackingEngine (video ↔ worker ↔ tracker ↔ overlay), CSV export
src/render/      canvas overlay
src/components/  React UI
```

## License

MIT for this code. The YOLOX models are Apache-2.0 (Megvii).
