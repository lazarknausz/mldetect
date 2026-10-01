import type { Affine } from '../vision/regionTracker';
import type { Homography, Point, Similarity } from '../vision/transforms';
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

/** Long side of the greyscale image the speed gun measures motion on (full 720p detail), px. */
export const GUN_MOTION_IMAGE_SIZE = 1280;

/** A box to follow from the previous frame of the sequence into this one (speed gun). */
export interface GunTarget {
  id: number;
  /** Box in the previous frame, source px. */
  box: { x1: number; y1: number; x2: number; y2: number };
  /** Expected shift since the previous frame (e.g. last frame's), source px. */
  guess: Point;
  /** Vehicle mode: the calibration line on the vehicle (its motion is fitted separately). */
  line?: { a: Point; b: Point };
}

/** One speed-gun step: optional detection + camera motion + optical-flow target tracking. */
export interface GunParams {
  bitmapScale: number;
  imageWidth: number;
  imageHeight: number;
  /** Run the detector on these regions this frame; null = tracking only. */
  detect: { regions: Region[]; scoreThreshold: number; classes: number[] | null } | null;
  /** A new sequence starts after a seek; frames are only compared within a sequence. */
  sequence: number;
  t: number;
  targets: GunTarget[];
}

export interface GunResponse {
  detections: Detection[];
  /** Camera motion previous → this frame (source px); undefined: no previous frame; null: unmeasurable. */
  camera?: Homography | null;
  /** Per target: its motion previous → this frame (source px), or null if it could not be followed. */
  targets: Array<{ id: number; motion: Similarity | null; affine: Affine | null; lineAffine: Affine | null; inliers: number }>;
  ms: number;
}

export type WorkerRequest =
  | { type: 'init'; modelUrl: string; inputSize: number; preferWebGPU: boolean }
  | ({ type: 'detect'; id: number; bitmap: ImageBitmap } & DetectParams)
  | ({ type: 'gun'; id: number; bitmap: ImageBitmap } & GunParams);

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
  | ({ type: 'result'; id: number } & DetectResponse)
  | ({ type: 'gunResult'; id: number } & GunResponse);
