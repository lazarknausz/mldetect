/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import { buildPyramid, estimateGlobalMotion, rgbaToGray, type GrayImage } from '../vision/globalMotion';
import { rescaleH } from '../vision/transforms';
import { NUM_CLASSES } from './classes';
import { decodeYolox } from './postprocess';
import { PAD_VALUE, rgbaToBgrChw } from './preprocess';
import { MOTION_IMAGE_SIZE, type DetectResponse, type WorkerRequest, type WorkerResponse } from './protocol';
import { dropCutAtTileEdge, mergeRegionDetections, regionRatio } from './tiling';
import type { Backend, Detection } from './types';

declare const self: DedicatedWorkerGlobalScope;

const IOU_THRESHOLD = 0.45;
/** Frames further apart than this are not compared for camera motion, seconds. */
const MAX_MOTION_GAP_SEC = 1;

let session: ort.InferenceSession | null = null;
let inputName = 'images';
let outputName = 'output';
let inputSize = 416;
let ctx: OffscreenCanvasRenderingContext2D;
let tensorData: Float32Array<ArrayBuffer>;
let motionCtx: OffscreenCanvasRenderingContext2D | null = null;
/** Previous frame for camera-motion measurement. */
let prevFrame: {
  sequence: number;
  t: number;
  pyramid: GrayImage[];
  /** Grey-image px per source px. */
  scale: number;
  detections: Detection[];
} | null = null;

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

async function downloadModel(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download model (HTTP ${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  let lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pct = total ? Math.floor((received / total) * 100) : -1;
    if (pct !== lastPct && pct % 5 === 0) {
      lastPct = pct;
      post({ type: 'progress', message: `Downloading model… ${pct}%` });
    }
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf;
}

async function runModel(): Promise<Float32Array> {
  const input = new ort.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]);
  const results = await session!.run({ [inputName]: input });
  const out = results[outputName];
  const data = (await out.getData()) as Float32Array;
  out.dispose?.();
  return data;
}

async function createSession(model: Uint8Array, backend: Backend) {
  const s = await ort.InferenceSession.create(model, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  });
  session = s;
  inputName = s.inputNames[0];
  outputName = s.outputNames[0];
  // Warm-up run (compiles shaders / allocates buffers) and sanity check.
  tensorData.fill(PAD_VALUE);
  const data = await runModel();
  if (!data.length || !Number.isFinite(data[4])) throw new Error('backend produced invalid output');
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${what} timed out`)), ms);
    p.then(
      (v) => (clearTimeout(timer), resolve(v)),
      (e) => (clearTimeout(timer), reject(e)),
    );
  });
}

async function hasWebGPUAdapter(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return !!(await withTimeout(gpu.requestAdapter(), 5000, 'WebGPU adapter request'));
  } catch {
    return false;
  }
}

async function init(modelUrl: string, size: number, preferWebGPU: boolean) {
  inputSize = size;
  ctx = new OffscreenCanvas(size, size).getContext('2d', { willReadFrequently: true })!;
  tensorData = new Float32Array(3 * size * size);
  ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.origin).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  const model = await downloadModel(modelUrl);
  if (preferWebGPU && (await hasWebGPUAdapter())) {
    try {
      post({ type: 'progress', message: 'Starting WebGPU backend…' });
      await withTimeout(createSession(model, 'webgpu'), 30000, 'WebGPU start-up');
      post({ type: 'ready', backend: 'webgpu' });
      return;
    } catch (err) {
      // A failed or hung WebGPU start can leave the runtime unusable: ask the main
      // thread for a fresh worker that goes straight to WASM.
      console.warn('[detector] WebGPU backend failed, falling back to WASM', err);
      post({ type: 'fallback' });
      return;
    }
  }
  post({ type: 'progress', message: 'Starting WASM backend…' });
  await createSession(model, 'wasm');
  post({ type: 'ready', backend: 'wasm' });
}

async function detectRegions(msg: Extract<WorkerRequest, { type: 'detect' }>): Promise<Detection[]> {
  const bs = msg.bitmapScale;
  let full: Detection[] = [];
  const tiles: Detection[] = [];
  for (let i = 0; i < msg.regions.length; i++) {
    const r = msg.regions[i];
    const ratio = regionRatio(r, inputSize);
    ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(msg.bitmap, r.x * bs, r.y * bs, r.w * bs, r.h * bs, 0, 0, r.w * ratio, r.h * ratio);
    const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
    rgbaToBgrChw(data, inputSize, tensorData);
    const output = await runModel();
    const dets = decodeYolox(output, {
      inputSize,
      numClasses: NUM_CLASSES,
      ratio,
      offsetX: r.x,
      offsetY: r.y,
      imageWidth: msg.imageWidth,
      imageHeight: msg.imageHeight,
      scoreThreshold: msg.scoreThreshold,
      iouThreshold: IOU_THRESHOLD,
      classFilter: msg.classes ? new Set(msg.classes) : null,
    });
    if (i === 0) full = dets;
    else tiles.push(...dropCutAtTileEdge(dets, r, msg.imageWidth, msg.imageHeight));
  }
  return msg.regions.length > 1 ? mergeRegionDetections(full, tiles, IOU_THRESHOLD) : full;
}

/** Camera motion since the previous frame of the same sequence (see DetectResponse.camera). */
function measureCamera(msg: Extract<WorkerRequest, { type: 'detect' }>, detections: Detection[]): DetectResponse['camera'] {
  if (!msg.motion) {
    prevFrame = null;
    return undefined;
  }
  const scale = Math.min(msg.bitmapScale, MOTION_IMAGE_SIZE / Math.max(msg.imageWidth, msg.imageHeight));
  const gw = Math.max(1, Math.round(msg.imageWidth * scale));
  const gh = Math.max(1, Math.round(msg.imageHeight * scale));
  if (!motionCtx || motionCtx.canvas.width !== gw || motionCtx.canvas.height !== gh) {
    motionCtx = new OffscreenCanvas(gw, gh).getContext('2d', { willReadFrequently: true })!;
  }
  motionCtx.drawImage(msg.bitmap, 0, 0, gw, gh);
  const pyramid = buildPyramid(rgbaToGray(motionCtx.getImageData(0, 0, gw, gh).data, gw, gh), 4);
  const prev = prevFrame;
  prevFrame = { sequence: msg.motion.sequence, t: msg.motion.t, pyramid, scale, detections };
  const dt = prev ? msg.motion.t - prev.t : 0;
  if (!prev || prev.sequence !== msg.motion.sequence || prev.scale !== scale || dt <= 0 || dt > MAX_MOTION_GAP_SEC) {
    return undefined;
  }
  // Keep features off everything that may move by itself.
  const exclude = prev.detections.map((d) => ({ x1: d.x1 * scale, y1: d.y1 * scale, x2: d.x2 * scale, y2: d.y2 * scale }));
  const m = estimateGlobalMotion(prev.pyramid, pyramid, exclude);
  return m ? rescaleH(m.transform, 1 / scale) : null;
}

async function detect(msg: Extract<WorkerRequest, { type: 'detect' }>) {
  if (!session) throw new Error('Model not loaded');
  const t0 = performance.now();
  try {
    const detections = await detectRegions(msg);
    const camera = measureCamera(msg, detections);
    post({ type: 'result', id: msg.id, detections, inferMs: performance.now() - t0, camera });
  } finally {
    msg.bitmap.close();
  }
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    init(msg.modelUrl, msg.inputSize, msg.preferWebGPU).catch((err) =>
      post({ type: 'error', message: `Failed to load model: ${String(err?.message ?? err)}` }),
    );
  } else if (msg.type === 'detect') {
    detect(msg).catch((err) => post({ type: 'error', id: msg.id, message: String(err?.message ?? err) }));
  }
};
