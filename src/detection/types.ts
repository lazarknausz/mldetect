/** A detected object in source-video pixel coordinates. */
export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
}

export type Backend = 'webgpu' | 'wasm';

export interface DetectorStatus {
  state: 'idle' | 'loading' | 'ready' | 'error';
  backend?: Backend;
  message?: string;
}
