/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import { NUM_CLASSES } from './classes';
import { decodeYolox } from './postprocess';
import { PAD_VALUE, rgbaToBgrChw } from './preprocess';
import type { WorkerRequest, WorkerResponse } from './protocol';
import type { Backend } from './types';

declare const self: DedicatedWorkerGlobalScope;

const IOU_THRESHOLD = 0.45;

let session: ort.InferenceSession | null = null;
let inputName = 'images';
let outputName = 'output';
let inputSize = 416;
let ctx: OffscreenCanvasRenderingContext2D;
let tensorData: Float32Array<ArrayBuffer>;

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

async function detect(msg: Extract<WorkerRequest, { type: 'detect' }>) {
  if (!session) throw new Error('Model not loaded');
  const t0 = performance.now();
  ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(msg.bitmap, 0, 0);
  msg.bitmap.close();
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);
  rgbaToBgrChw(data, inputSize, tensorData);
  const output = await runModel();
  const detections = decodeYolox(output, {
    inputSize,
    numClasses: NUM_CLASSES,
    ratio: msg.ratio,
    imageWidth: msg.imageWidth,
    imageHeight: msg.imageHeight,
    scoreThreshold: msg.scoreThreshold,
    iouThreshold: IOU_THRESHOLD,
    classFilter: msg.classes ? new Set(msg.classes) : null,
  });
  post({ type: 'result', id: msg.id, detections, inferMs: performance.now() - t0 });
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    init(msg.modelUrl, msg.inputSize, msg.preferWebGPU).catch((err) =>
      post({ type: 'error', message: `Failed to load model: ${String(err?.message ?? err)}` }),
    );
  } else if (msg.type === 'detect') {
    detect(msg).catch((err) => {
      msg.bitmap.close();
      post({ type: 'error', id: msg.id, message: String(err?.message ?? err) });
    });
  }
};
