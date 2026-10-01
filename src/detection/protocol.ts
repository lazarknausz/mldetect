import type { Homography } from '../vision/transforms';
import type { Region } from './tiling';
import type { Backend, Detection } from './types';

/** Long side of the greyscale image the worker measures camera motion on, px. */
export const MOTION_IMAGE_SIZE = 640;

export interface DetectParams {
  /** The bitmap is the source frame resized by this factor. */
  bitmapScale: number;
  imageWidth: number;
  imageHeight: number;
  /** Crops of the source frame (source px) to run the detector on: the full frame, then tiles. */
  regions: Region[];
  scoreThreshold: number;
  classes: number[] | null;
  /**
   * Measure camera motion since the previous frame of the same `sequence` (a new
   * sequence starts after a seek / reset). `t` is the frame's media time, seconds.
   */
  motion: { sequence: number; t: number } | null;
}

export type WorkerRequest =
  | { type: 'init'; modelUrl: string; inputSize: number; preferWebGPU: boolean }
  | ({ type: 'detect'; id: number; bitmap: ImageBitmap } & DetectParams);

export interface DetectResponse {
  detections: Detection[];
  inferMs: number;
  /**
   * Camera motion from the previous frame to this one (source px): undefined when
   * there is no previous frame to compare with, null when it could not be measured
   * (too little background detail, e.g. clear sky).
   */
  camera?: Homography | null;
}

export type WorkerResponse =
  | { type: 'progress'; message: string }
  | { type: 'ready'; backend: Backend }
  /** WebGPU failed; the client should restart the worker in WASM-only mode. */
  | { type: 'fallback' }
  | { type: 'error'; message: string; id?: number }
  | ({ type: 'result'; id: number } & DetectResponse);
