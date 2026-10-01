/**
 * Speed Gun page — wiring.
 *
 *   empty ──(choose file)──► calibrate ──(line + metres, Start)──► measuring ──► done
 *                               ▲                                               │
 *                               └────────────────(redraw / Start again)─────────┘
 *
 * Modules:
 *   fileHandling.ts  load the file, detect FPS, seek to exact frames
 *   canvasView.ts    canvas setup, pointer → video coordinates, drawing
 *   calibration.ts   the calibration line + its real length (pixels → metres)
 *   analysisLoop.ts  frame-by-frame detection / optical-flow tracking loop
 *   meter.ts         per-vehicle distance and speed bookkeeping
 *   speedMath.ts     the distance-over-time formulas
 */

import { DetectorClient } from '../detection/detectorClient';
import { MODELS, type ModelId } from '../detection/models';
import { planRegions } from '../detection/tiling';
import type { DetectorStatus } from '../detection/types';
import { AnalysisLoop } from './analysisLoop';
import { CalibrationState } from './calibration';
import { CanvasView, GUN_CLASSES } from './canvasView';
import { detectVideoFps, frameCount, loadVideoFile, seekToFrame } from './fileHandling';
import type { CalibrationMode, SpeedMeter } from './meter';
import { frameIndexAt } from './speedMath';
import './speedgun.css';

// ------------------------------------------------------------------- elements

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const canvas = $<HTMLCanvasElement>('view');
const canvasMessage = $<HTMLDivElement>('canvasMessage');
const fileInput = $<HTMLInputElement>('fileInput');
const fileName = $<HTMLParagraphElement>('fileName');
const metresInput = $<HTMLInputElement>('metresInput');
const presetSelect = $<HTMLSelectElement>('presetSelect');
const modeField = $<HTMLFieldSetElement>('modeField');
const clearLine = $<HTMLButtonElement>('clearLine');
const modelSelect = $<HTMLSelectElement>('modelSelect');
const fpsInput = $<HTMLInputElement>('fpsInput');
const startBtn = $<HTMLButtonElement>('startBtn');
const stopBtn = $<HTMLButtonElement>('stopBtn');
const progressBar = $<HTMLDivElement>('progressBar').firstElementChild as HTMLDivElement;
const perfLabel = $<HTMLParagraphElement>('perfLabel');
const currentSpeed = $<HTMLSpanElement>('currentSpeed');
const maxSpeed = $<HTMLSpanElement>('maxSpeed');
const ratioLabel = $<HTMLSpanElement>('ratioLabel');
const fpsLabel = $<HTMLSpanElement>('fpsLabel');
const results = $<HTMLDivElement>('results');
const modelDot = $<HTMLSpanElement>('modelDot');
const modelStatus = $<HTMLSpanElement>('modelStatus');
const scrub = $<HTMLInputElement>('scrub');
const frameLabel = $<HTMLSpanElement>('frameLabel');
const firstFrameBtn = $<HTMLButtonElement>('firstFrame');
const prevFrameBtn = $<HTMLButtonElement>('prevFrame');
const nextFrameBtn = $<HTMLButtonElement>('nextFrame');

// ---------------------------------------------------------------------- state

type Phase = 'empty' | 'calibrate' | 'measuring' | 'done';
let phase: Phase = 'empty';
let fps = 30;
let totalFrames = 1;
let frame = 0;
let loop: AnalysisLoop | null = null;
let lastMeter: SpeedMeter | null = null;
let detector: DetectorClient;
let detectorReady = false;
let userPickedMode = false;

const view = new CanvasView(canvas);
const calibration = new CalibrationState(canvas, view.toVideo, view.videoPerScreen, onCalibrationChange, () => frame);

// ----------------------------------------------------------------- detector

function startDetector(id: ModelId) {
  detector?.terminate();
  detectorReady = false;
  const spec = MODELS[id];
  detector = new DetectorClient(spec, `${import.meta.env.BASE_URL}models/${spec.file}`, (s: DetectorStatus) => {
    detectorReady = s.state === 'ready';
    modelDot.className = `dot ${s.state}`;
    modelStatus.textContent =
      s.state === 'ready'
        ? `Detector ready · ${s.backend === 'webgpu' ? 'GPU (WebGPU)' : 'CPU (WASM)'}`
        : s.state === 'error'
          ? `Detector error: ${s.message}`
          : (s.message ?? 'Loading detector…');
    updateControls();
  });
}
startDetector(modelSelect.value as ModelId);
modelSelect.addEventListener('change', () => startDetector(modelSelect.value as ModelId));

// ------------------------------------------------------------ file handling

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  loop?.stop();
  try {
    canvasMessage.hidden = false;
    canvasMessage.innerHTML = '<p>Loading video…</p>';
    const info = await loadVideoFile(file, video);
    view.setSize(info.width, info.height);
    fps = await detectVideoFps(video);
    totalFrames = frameCount(video, fps);
    fpsInput.value = String(fps);
    fileName.textContent = `${info.name} · ${info.width}×${info.height} · ${info.duration.toFixed(1)} s`;
    scrub.max = String(totalFrames - 1);
    calibration.clear();
    calibration.metres = null;
    metresInput.value = '';
    presetSelect.value = '';
    userPickedMode = false;
    resetReadouts();
    phase = 'calibrate';
    await showFrame(0);
    canvasMessage.hidden = true;
  } catch (err) {
    phase = 'empty';
    canvasMessage.hidden = false;
    canvasMessage.innerHTML = `<p>${(err as Error).message}</p>`;
  }
  updateControls();
});

fpsInput.addEventListener('change', () => {
  const v = Number(fpsInput.value);
  if (v > 0) {
    fps = v;
    totalFrames = frameCount(video, fps);
    scrub.max = String(totalFrames - 1);
    updateControls();
  }
});

// ------------------------------------------------------------ frame display

async function showFrame(n: number) {
  frame = Math.max(0, Math.min(totalFrames - 1, n));
  const shown = await seekToFrame(video, frame, fps);
  frame = shown;
  redraw();
}

/** Draws the current frame and everything on top of it. */
function redraw(meter: SpeedMeter | null = lastMeter) {
  if (phase === 'empty') return;
  view.drawFrame(video);
  const measuring = phase === 'measuring' || phase === 'done';
  const cal = calibration.complete ? calibration.toCalibration() : null;
  // Road mode keeps showing the line and its zone; vehicle mode only before measuring.
  if (!measuring || cal?.mode === 'ground') view.drawCalibration(calibration.a, calibration.b, calibration.metres, cal);
  if (meter && measuring) view.drawTracks(meter);
  scrub.value = String(frame);
  frameLabel.textContent = `frame ${frame} / ${totalFrames - 1} · ${(frame / fps).toFixed(2)} s`;
}

scrub.addEventListener('input', () => void showFrame(Number(scrub.value)));
firstFrameBtn.addEventListener('click', () => void showFrame(0));
prevFrameBtn.addEventListener('click', () => void showFrame(frame - 1));
nextFrameBtn.addEventListener('click', () => void showFrame(frame + 1));

// -------------------------------------------------------------- calibration

function onCalibrationChange() {
  if (phase === 'done') {
    phase = 'calibrate';
    lastMeter = null;
  }
  redraw();
  updateControls();
  if (calibration.lengthPx >= 10) void suggestMode();
}

metresInput.addEventListener('input', () => {
  const v = Number(metresInput.value);
  calibration.metres = v > 0 ? v : null;
  presetSelect.value = '';
  redraw();
  updateControls();
});

presetSelect.addEventListener('change', () => {
  if (!presetSelect.value) return;
  metresInput.value = presetSelect.value;
  calibration.metres = Number(presetSelect.value);
  // Lengths of vehicles imply "the vehicle itself".
  const opt = presetSelect.selectedOptions[0].textContent ?? '';
  if (/Boeing|Airbus|car length/i.test(opt)) setMode('object');
  else setMode('ground');
  redraw();
  updateControls();
});

modeField.addEventListener('change', () => {
  userPickedMode = true;
  calibration.mode = (modeField.querySelector('input:checked') as HTMLInputElement).value as CalibrationMode;
  redraw();
  updateControls();
});

function setMode(mode: CalibrationMode) {
  calibration.mode = mode;
  (modeField.querySelector(`input[value="${mode}"]`) as HTMLInputElement).checked = true;
}

clearLine.addEventListener('click', () => calibration.clear());

let suggesting = false;
/**
 * If the line lies along a detected vehicle and is about as long as it, the user most
 * likely measured the vehicle itself: suggest "vehicle" mode (they can change it).
 */
async function suggestMode() {
  if (suggesting || userPickedMode || !detectorReady || !calibration.a || !calibration.b) return;
  suggesting = true;
  try {
    const bitmap = await createImageBitmap(video);
    const W = video.videoWidth;
    const H = video.videoHeight;
    const res = await detector.detect(bitmap, {
      bitmapScale: 1,
      imageWidth: W,
      imageHeight: H,
      regions: planRegions(W, H, detector.model.inputSize, 'standard'),
      scoreThreshold: 0.3,
      classes: GUN_CLASSES.object,
      motion: null,
    });
    const a = calibration.a!;
    const b = calibration.b!;
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const onVehicle = res.detections.some((d) => {
      const inside = mid.x >= d.x1 && mid.x <= d.x2 && mid.y >= d.y1 && mid.y <= d.y2;
      const extent = (Math.abs(b.x - a.x) / len) * (d.x2 - d.x1) + (Math.abs(b.y - a.y) / len) * (d.y2 - d.y1);
      return inside && len > 0.5 * extent && len < 1.5 * extent;
    });
    if (!userPickedMode) setMode(onVehicle ? 'object' : 'ground');
    redraw();
    updateControls();
  } catch {
    // Suggestion only; ignore failures.
  } finally {
    suggesting = false;
  }
}

// ------------------------------------------------------------------ measuring

startBtn.addEventListener('click', async () => {
  if (!calibration.complete || !detectorReady) return;
  const cal = calibration.toCalibration();
  phase = 'measuring';
  resetReadouts();
  updateControls();
  const startFrame = calibration.frame;
  const started = performance.now();
  let processed = 0;
  let workerMs = 0;
  loop = new AnalysisLoop(
    video,
    detector,
    cal,
    {
      fps,
      startFrame,
      endFrame: totalFrames - 1,
      detectEvery: cal.mode === 'ground' ? 3 : 4,
      classes: GUN_CLASSES[cal.mode],
      scoreThreshold: 0.3,
    },
    {
      onFrame(n, meter, res) {
        workerMs = 0.9 * workerMs + 0.1 * res.ms;
        frame = n;
        lastMeter = meter;
        processed++;
        redraw(meter);
        updateReadouts(meter);
        progressBar.style.width = `${(((n - startFrame + 1) / (totalFrames - startFrame)) * 100).toFixed(1)}%`;
        const rate = processed / ((performance.now() - started) / 1000);
        perfLabel.textContent = `Analysing frame ${n} of ${totalFrames - 1} · ${rate.toFixed(1)} frames/s · worker ${workerMs.toFixed(0)} ms/frame`;
      },
      onDone(meter, stopped) {
        phase = 'done';
        lastMeter = meter;
        perfLabel.textContent = stopped ? 'Stopped.' : 'Finished: every frame analysed.';
        updateReadouts(meter);
        updateControls();
      },
      onError(err) {
        phase = 'done';
        perfLabel.textContent = `Error: ${err.message}`;
        updateControls();
      },
    },
  );
  await loop.run();
});

stopBtn.addEventListener('click', () => loop?.stop());

// ------------------------------------------------------------------ read-outs

function resetReadouts() {
  currentSpeed.textContent = '–';
  maxSpeed.textContent = '–';
  results.innerHTML = '';
  progressBar.style.width = '0';
  perfLabel.textContent = '';
}

function fmt(v: number | null): string {
  return v === null ? '–' : v.toFixed(0);
}

function updateReadouts(meter: SpeedMeter) {
  currentSpeed.textContent = fmt(meter.currentKmh);
  maxSpeed.textContent = fmt(meter.maxKmh);
  if (meter.cal.mode !== 'ground') return;
  // Road mode: list every vehicle measured in the zone.
  const done = [...meter.retired, ...meter.tracks].filter((t) => t.status === 'measured').sort((p, q) => p.id - q.id);
  if (!done.length) return;
  const rows = done
    .map(
      (t) =>
        `<tr><td>#${t.id}</td><td>${t.fullCrossing ? '' : '≈ '}${fmt(t.finalKmh)} km/h</td><td>${
          t.fullCrossing
            ? 'full crossing: line length ÷ crossing time'
            : 'partial: seen in part of the zone only, less accurate'
        }</td></tr>`,
    )
    .join('');
  results.innerHTML = `<table><thead><tr><th>Vehicle</th><th>Speed</th><th>Method</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function updateControls() {
  const hasVideo = phase !== 'empty';
  const running = phase === 'measuring';
  calibration.enabled = hasVideo && !running;
  canvas.classList.toggle('drawing', calibration.enabled);
  metresInput.disabled = !hasVideo || running;
  presetSelect.disabled = !hasVideo || running;
  modeField.disabled = !hasVideo || running;
  clearLine.disabled = !hasVideo || running || !calibration.a;
  fpsInput.disabled = !hasVideo || running;
  modelSelect.disabled = running;
  fileInput.disabled = running;
  for (const b of [firstFrameBtn, prevFrameBtn, nextFrameBtn]) b.disabled = !hasVideo || running;
  scrub.disabled = !hasVideo || running;
  startBtn.disabled = !hasVideo || running || !calibration.complete || !detectorReady;
  startBtn.textContent = phase === 'done' ? 'Measure again' : 'Start measuring';
  stopBtn.disabled = !running;
  const r = calibration.ratio;
  ratioLabel.textContent = r ? `1 px = ${r.toFixed(4)} m · ${calibration.mode === 'ground' ? 'road' : 'vehicle'}` : '–';
  fpsLabel.textContent = hasVideo ? `${fps} fps · 1 frame = ${(1000 / fps).toFixed(1)} ms` : '–';
  if (hasVideo && !running && !calibration.a) {
    canvasMessage.hidden = false;
    canvasMessage.innerHTML =
      '<p><b>Step 2.</b> Click and drag along something whose length you know,<br>e.g. a lane marking or the vehicle itself.</p>';
  } else if (hasVideo) {
    canvasMessage.hidden = true;
  }
}

// Keep the frame label right if someone plays the hidden video via devtools etc.
video.addEventListener('seeked', () => {
  if (phase !== 'measuring') frame = frameIndexAt(video.currentTime, fps);
});

updateControls();
