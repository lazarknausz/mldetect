import type { Backend, Detection } from './types';

export interface DetectParams {
  /** Letterbox scale: the bitmap was resized to (imageWidth × ratio, imageHeight × ratio). */
  ratio: number;
  imageWidth: number;
  imageHeight: number;
  scoreThreshold: number;
  classes: number[] | null;
}

export type WorkerRequest =
  | { type: 'init'; modelUrl: string; inputSize: number; preferWebGPU: boolean }
  | ({ type: 'detect'; id: number; bitmap: ImageBitmap } & DetectParams);

export type WorkerResponse =
  | { type: 'progress'; message: string }
  | { type: 'ready'; backend: Backend }
  /** WebGPU failed; the client should restart the worker in WASM-only mode. */
  | { type: 'fallback' }
  | { type: 'error'; message: string; id?: number }
  | { type: 'result'; id: number; detections: Detection[]; inferMs: number };
