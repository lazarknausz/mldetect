# MLDetect

A website that tracks every moving object in a video (cars on a highway, planes in the sky, boats, birds, cyclists…) and shows, for each one:

- a **bounding box** from the moment it appears, with a stable **ID** for as long as it stays in view;
- its estimated **speed in km/h**, measured relative to the scene even when the camera pans or zooms to follow it;
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
- **Model**: *Fast* (YOLOX-Tiny, 416 px) or *Accurate* (YOLOX-S, 640 px, about 3× slower).
- **Small / distant objects**: also runs the detector on overlapping zoomed-in tiles of the frame, so cars in the background are found too. *On* (default) uses 2–4 tiles (≈ 3× the work), *Maximum* 6–9 tiles; *Off* is fastest.
- **Objects**: road vehicles, aircraft & birds, anything that moves, or all 80 COCO classes.
- **Speed calibration**: the *camera zoom* used for the video (phones: 1× = main lens) and the *aircraft type* (a Cessna and an A350 are both just "airplane" to the detector).
- The panel shows whether the camera is *steady*, *moving · compensated*, or *unmeasurable* (no background detail, e.g. only clear sky: speeds may then be off).
- Click a box or a table row to highlight one object.

## How it works

```
<video> frame ─► Web Worker: full frame + zoomed tiles → YOLOX (ONNX Runtime Web, WebGPU → WASM)
             │                → decode + NMS, merge tiles
             │   camera motion: background corners → Lucas–Kanade flow → RANSAC → homography
             ─► Tracker: camera-compensated Kalman filter per object + ByteTrack association
             ─► Motion: 3-D speed from a pinhole camera model + each class's real size
             ─► Canvas overlay (boxes, trails, arrows, predicted paths) + live table
```

- **Detection**: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) COCO models (Apache-2.0), run with ONNX Runtime Web in a Web Worker. It uses WebGPU when available and falls back to multi-threaded WASM. The model sees a fixed 416 / 640 px square, so a 1080p frame is shrunk ~4.6× and a background car becomes a few pixels; **sliced inference** (`src/detection/tiling.ts`) additionally runs it on overlapping tiles at 2–3× the magnification, drops boxes cut by a tile border and merges the rest with the full-frame result.
- **Camera motion** (`src/vision/`): corners on the background (outside every detected object) are followed from frame to frame with pyramidal Lucas–Kanade optical flow; RANSAC finds the camera's motion among them and it is refined to a homography with the keystone of a turning camera. Kalman filters are warped by it and track histories are stored in a scene-fixed frame, so an object's motion is measured relative to the scene, not the screen: a plane the camera follows is not "standing still", and a parked car the camera pans past is not "moving".
- **Tracking** (`src/tracking/tracker.ts`): constant-velocity Kalman filter per box. ByteTrack-style matching first pairs confident detections with predicted boxes (IoU, plus a motion-aware distance gate for fast objects at low frame rates), then uses low-confidence detections to keep existing tracks alive. Tracks survive about 1 s of occlusion and are dropped once they leave the frame.
- **Speed** (`src/tracking/motion.ts`): each class has a real-world size as a length × width × height block (car 4.5 × 1.8 × 1.5 m, airliner 38 × 35 × 12 m, …). With a pinhole camera model, the box of such a block depends on its distance, its 3-D direction of travel and the angle it is seen from (side-on, from below, from an overpass…), so the estimator searches for the 3-D motion that explains both the on-screen motion and how the box grows, shrinks and changes shape over the last 2 s. Motion towards / away from the camera is only accepted when the boxes show it beyond detector noise. Readings are median-filtered and smoothed; brand-new tracks show *measuring…*.
- **Prediction**: constant speed + constant turn rate (CTRV) extrapolation of the object's own (camera-compensated) motion.

### Accuracy and limitations

In simulated 3-D scenes with detector-like box jitter (`src/tracking/simulation.test.ts`: cars crossing, approaching an overpass, driving away, a 20 px background car, airliners overhead, side-on through a 3× zoom, climbing away, and a camera panning to follow a plane), readings settle within **±5–15 %** of the true speed — provided the object's real size and the camera zoom match the settings:

- **Real size is the main error source.** Speed scales with the assumed size: a 4.2 m hatchback reads 7 % fast at the default 4.5 m; COCO "truck" covers vans to lorries. Set the aircraft type for planes.
- **Set the camera zoom.** It matters for motion towards / away from the camera.
- **Panning, tilting and zooming are compensated** as long as some background detail (ground, buildings, trees, clouds) is visible. In a clear sky there is nothing to measure the camera against, and the panel says so. A camera that *travels* (dashcam, drone flying along) is not compensated: speeds are then relative to the camera.
- Motion mostly towards or away from the camera needs 1–2 s of track history to settle.
- Objects partly outside the frame keep their last reading until they are fully visible.

## Speed gun (calibrated)

`speed-gun.html` (linked from the main page) measures speed the way a LIDAR speed gun does, **distance over time**, from a calibration line you draw on the video:

1. **Load** an .mp4 / .webm. The frame rate is detected automatically (it can be edited).
2. **Calibrate:** drag a line over something of known length and type the length (presets: US lane-marking cycle 12.19 m, car 4.5 m, Boeing 737-800 39.5 m, …). This gives *metres per pixel*.
3. **Measure:** the gun steps through **every frame**. Time between frames is exactly 1 / FPS. Distance is the vehicle's pixel displacement × metres per pixel. Speed = distance / time, shown on the box, as *Current speed* and as *Max speed detected*.

Two kinds of calibration, because one metres-per-pixel ratio is only true at one distance from the camera:

| Mode | Line drawn on | How it measures | Best for |
| --- | --- | --- | --- |
| **Road** (speed trap) | lane markings, along the road | Each vehicle's ground point (bottom of its box) is timed across the line's stretch: **line length ÷ crossing time**. The image row of a road point depends only on its distance down the road, so one lane's markings calibrate every lane. | fixed cameras over a road |
| **Vehicle** | the vehicle itself, nose to tail | The line is carried along on the vehicle frame by frame, so the scale follows it as it comes closer, turns or the camera zooms. Displacement is measured against the background, so a panning camera is compensated. | aircraft, a followed car |

Precision comes from: sub-pixel Lucas–Kanade optical flow on the vehicle's own texture (not jittery detector boxes); camera motion measured from the background with RANSAC; and least-squares fits over many frames instead of single-frame differences.

**Validation on real footage** (`4K Video of Highway Traffic`, overpass camera, road mode, calibrated on one 12.19 m lane-marking cycle): over the first 10 s, 21 cars were also timed by an independent method (raw per-frame detector boxes crossing the same two rows, no optical flow). Median difference **1.3 %**, mean +0.9 %, worst 6 %. Simulated 3-D scenes (perspective highway, panning and zooming camera following an airliner) are in `src/speedgun/meter.test.ts` and land within 1.5–2 %.

**Limits:** one camera only sees motion across the picture. A vehicle driving straight at the camera barely moves on screen, so its speed can't be measured then (like a radar gun's cosine error, the other way round). The vehicle-mode reading is held while the vehicle is partly out of the frame. Accuracy is only as good as the calibration line: a 2 px error on a 100 px line is 2 % in speed.

## Deploying

The site is published to **GitHub Pages** at <https://lazarknausz.github.io/mldetect/> from the `gh-pages` branch. `.github/workflows/deploy.yml` rebuilds and republishes it on every push to `main`.

One-time setup: *Settings → Pages → Build and deployment → Source: "Deploy from a branch"*, branch **gh-pages**, folder **/ (root)**. On a free GitHub plan, Pages only works for **public** repositories.

The site is fully static, so any static host works: serve `dist/`, ideally with these headers to enable multi-threaded WASM:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them (e.g. on GitHub Pages) it still works: WebGPU is unaffected and the CPU fallback runs single-threaded.

## Project layout

```
src/detection/   model registry, worker, pre/post-processing (decode, NMS), tiling, class sizes
src/vision/      camera motion (optical flow, RANSAC, homographies), region tracker
src/tracking/    Kalman filter, Hungarian assignment, ByteTrack tracker, motion & speed
src/engine/      TrackingEngine (video ↔ worker ↔ tracker ↔ overlay), CSV export
src/render/      canvas overlay
src/components/  React UI
src/speedgun/    speed gun page: speedMath (formulas), meter (trap / vehicle logic),
                 analysisLoop (frame stepping), calibration, canvasView, fileHandling, main
```

## License

MIT for this code. The YOLOX models are Apache-2.0 (Megvii).
