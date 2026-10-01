/**
 * The detection / tracking loop. Unlike normal playback (where the browser may skip
 * frames when it is busy), the gun visits *every* frame in order:
 *
 *   for frame n = start … end:
 *     1. seek the video to frame n                     → time t = n / FPS
 *     2. send the frame to the worker, which
 *          - every few frames runs the object detector (finds new vehicles),
 *          - measures how the camera moved since frame n − 1,
 *          - follows each tracked vehicle with sub-pixel optical flow;
 *     3. the SpeedMeter turns those pixel displacements into metres and fits
 *        distance against time → speed.
 */

import type { DetectorClient } from '../detection/detectorClient';
import type { GunResponse } from '../detection/protocol';
import type { Region } from '../detection/tiling';
import { centre, SpeedMeter, type Calibration } from './meter';
import { seekToFrame } from './fileHandling';
import { frameInterval } from './speedMath';

export interface LoopOptions {
  fps: number;
  startFrame: number;
  endFrame: number;
  /** Run the detector every N frames (optical flow runs on every frame). */
  detectEvery: number;
  classes: number[] | null;
  scoreThreshold: number;
}

export interface LoopCallbacks {
  onFrame(n: number, meter: SpeedMeter, res: GunResponse): void;
  onDone(meter: SpeedMeter, stopped: boolean): void;
  onError(err: Error): void;
}

let sequenceCounter = 1;

export class AnalysisLoop {
  readonly meter: SpeedMeter;
  private stopped = false;
  private readonly W: number;
  private readonly H: number;

  constructor(
    private video: HTMLVideoElement,
    private detector: DetectorClient,
    private cal: Calibration,
    private opts: LoopOptions,
    private cb: LoopCallbacks,
  ) {
    this.W = video.videoWidth;
    this.H = video.videoHeight;
    this.meter = new SpeedMeter(cal, this.W, this.H);
  }

  stop() {
    this.stopped = true;
  }

  async run(): Promise<void> {
    const { fps, startFrame, endFrame, detectEvery } = this.opts;
    const sequence = sequenceCounter++;
    let lastShown = -1;
    try {
      for (let n = startFrame; n <= endFrame && !this.stopped; n++) {
        const shown = await seekToFrame(this.video, n, fps);
        if (shown === lastShown) continue; // decoder returned the same frame again
        lastShown = shown;

        // Time of this frame: frames since the start × (1 / FPS).
        const t = (n - startFrame) * frameInterval(fps);
        const first = n === startFrame;
        const runDetector = first || (n - startFrame) % detectEvery === 0 || this.meter.needsDetection();
        const bitmap = await createImageBitmap(this.video);
        const res = await this.detector.gunStep(bitmap, {
          bitmapScale: 1,
          imageWidth: this.W,
          imageHeight: this.H,
          detect: runDetector
            ? { regions: this.regions(first), scoreThreshold: this.opts.scoreThreshold, classes: this.opts.classes }
            : null,
          sequence,
          t,
          targets: this.meter.targets(),
        });
        if (first && this.cal.mode === 'object') this.meter.initObject(res.detections);
        this.meter.update({
          t,
          camera: first ? undefined : res.camera,
          motions: new Map(res.targets.map((x) => [x.id, x.motion])),
          affines: new Map(res.targets.map((x) => [x.id, x.affine])),
          lineAffines: new Map(res.targets.map((x) => [x.id, x.lineAffine])),
          detections: runDetector && !(first && this.cal.mode === 'object') ? res.detections : undefined,
        });
        this.cb.onFrame(n, this.meter, res);
      }
      this.cb.onDone(this.meter, this.stopped);
    } catch (err) {
      this.cb.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }

  /**
   * Where to run the detector. Road mode: the whole frame — vehicles must be found
   * before they reach the zone, and a crop could cut a box's bottom edge (the ground
   * point we measure). Vehicle mode: a crop around the vehicle, which also magnifies a
   * distant plane so the detector sees it better.
   */
  private regions(first: boolean): Region[] {
    const W = this.W;
    const H = this.H;
    const clip = (x1: number, y1: number, x2: number, y2: number): Region => {
      const x = Math.max(0, Math.floor(x1));
      const y = Math.max(0, Math.floor(y1));
      return { x, y, w: Math.min(W, Math.ceil(x2)) - x, h: Math.min(H, Math.ceil(y2)) - y };
    };
    if (this.cal.mode === 'ground') return [{ x: 0, y: 0, w: W, h: H }];
    // Vehicle mode: around the line (first frame) or the followed vehicle.
    const tr = this.meter.tracks[0];
    const c = tr ? centre(tr.box) : { x: (this.cal.a.x + this.cal.b.x) / 2, y: (this.cal.a.y + this.cal.b.y) / 2 };
    const size = tr
      ? Math.max(tr.box.x2 - tr.box.x1, tr.box.y2 - tr.box.y1)
      : Math.hypot(this.cal.b.x - this.cal.a.x, this.cal.b.y - this.cal.a.y);
    const half = Math.max(200, 1.6 * size);
    const crop = clip(c.x - half, c.y - half, c.x + half, c.y + half);
    // On the first frame also look at the whole picture, in case the line is off.
    return first ? [crop, { x: 0, y: 0, w: W, h: H }] : [crop];
  }
}
