import { MAX_PLAUSIBLE_KMH, REFERENCE_SIZE, className, sameClassGroup, type ReferenceSize } from '../detection/classes';
import { iou } from '../detection/postprocess';
import type { Detection } from '../detection/types';
import { assign } from './assignment';
import { BoxKalman, type Box } from './kalman';
import {
  H_IDENTITY,
  applyH,
  applyHLinear,
  composeH,
  cornerMotion,
  invertH,
  rotationFromH,
  scaleMotionH,
  transposeMul,
  type Homography,
} from '../vision/transforms';
import {
  MS_TO_KMH,
  compassLabel,
  estimateSpeed,
  estimateTurnRate,
  fitVelocity,
  focalLengthPx,
  headingDegrees,
  median,
  zoomToDiagonalFov,
  type SpeedEstimate,
} from './motion';
import type { CameraState, HistoryPoint, TrackSnapshot, TrackState } from './types';

export interface TrackerOptions {
  /** Detections at or above this score are "high confidence" (ByteTrack stage 1). */
  highThreshold: number;
  /** Detections between low and high are only used to extend existing tracks (stage 2). */
  lowThreshold: number;
  /** Matched frames needed before a new track is shown. */
  minHits: number;
  /** How long a track survives without detections (occlusion), seconds. */
  maxLostSec: number;
  /** Window for the least-squares velocity fit, seconds. */
  velocityWindowSec: number;
  /** Window for the turn-rate estimate, seconds. */
  turnWindowSec: number;
  /** Length of the trail kept for drawing, seconds. */
  trailSec: number;
  /** Camera zoom relative to a typical phone main lens (sets the assumed field of view). */
  zoom: number;
  /** Per-class size overrides (e.g. the kind of aircraft). */
  referenceSizes: Partial<Record<number, ReferenceSize>>;
}

export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
  highThreshold: 0.4,
  lowThreshold: 0.1,
  minHits: 2,
  maxLostSec: 1.0,
  velocityWindowSec: 0.6,
  turnWindowSec: 1.2,
  trailSec: 3,
  zoom: 1,
  referenceSizes: {},
};

const MAX_HISTORY = 400;
/** Raw speed estimates are median-filtered over this window (rejects outliers)… */
const SPEED_MEDIAN_SEC = 1.0;
/** …and then exponentially smoothed with this time constant. */
const SPEED_SMOOTHING_SEC = 0.35;
/** Track history needed before a speed is shown, seconds. */
const MIN_SPEED_SPAN_SEC = 0.3;
/** Camera motion is extrapolated over gaps in the background (e.g. blur) this long. */
const MAX_CAMERA_GAP_SEC = 0.5;
/** The full 3-D speed model is re-run at most this often per track (it is costly). */
const SPEED_UPDATE_SEC = 0.1;
const DUPLICATE_IOU = 0.6;
const INFEASIBLE = Number.POSITIVE_INFINITY;

function toBox(d: Detection): Box {
  return { cx: (d.x1 + d.x2) / 2, cy: (d.y1 + d.y2) / 2, w: d.x2 - d.x1, h: d.y2 - d.y1 };
}

function toDet(b: Box): Detection {
  return { x1: b.cx - b.w / 2, y1: b.cy - b.h / 2, x2: b.cx + b.w / 2, y2: b.cy + b.h / 2, score: 0, classId: -1 };
}

class Track {
  readonly kf: BoxKalman;
  readonly history: HistoryPoint[] = [];
  readonly classVotes = new Map<number, number>();
  classId: number;
  score: number;
  state: TrackState = 'tentative';
  hits = 1;
  misses = 0;
  lastSeen: number;
  lastVelocity = { vx: 0, vy: 0 };
  measuring = true;
  /** Recent raw speed estimates, km/h. */
  speedSamples: Array<{ t: number; v: number }> = [];
  /** Exponentially smoothed speed for display, km/h. */
  speedEma: number | null = null;
  speedEmaT = 0;
  lastTurnRate = 0;
  /** Latest 3-D speed model result and when it was computed. */
  lastEstimate: { t: number; est: SpeedEstimate | null } | null = null;

  constructor(
    readonly id: number,
    det: Detection,
    readonly firstSeen: number,
  ) {
    this.kf = new BoxKalman(toBox(det));
    this.classId = det.classId;
    this.score = det.score;
    this.lastSeen = firstSeen;
  }

  vote(classId: number, score: number): void {
    const v = (this.classVotes.get(classId) ?? 0) + score;
    this.classVotes.set(classId, v);
    let best = this.classId;
    let bestV = -1;
    for (const [c, w] of this.classVotes) {
      if (w > bestV) {
        bestV = w;
        best = c;
      }
    }
    this.classId = best;
  }
}

export class Tracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private lastT: number | null = null;
  private frameWidth = 1;
  private frameHeight = 1;
  private _totalConfirmed = 0;
  /** Maps current-frame pixels to the scene-fixed reference frame (camera pose). */
  private pose: Homography = H_IDENTITY;
  /** Last measured camera motion (for extrapolation and screen velocity). */
  private lastCamera: { transform: Homography; dt: number; t: number } | null = null;
  private _cameraState: CameraState = 'static';
  opts: TrackerOptions;

  constructor(opts: Partial<TrackerOptions> = {}) {
    this.opts = { ...DEFAULT_TRACKER_OPTIONS, ...opts };
  }

  /** Number of distinct objects that have been confirmed since the last reset. */
  get totalConfirmed(): number {
    return this._totalConfirmed;
  }

  get lastTime(): number | null {
    return this.lastT;
  }

  /** Whether the camera is still, moving (compensated) or unmeasurable right now. */
  get cameraState(): CameraState {
    return this._cameraState;
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
    this.lastT = null;
    this._totalConfirmed = 0;
    this.pose = H_IDENTITY;
    this.lastCamera = null;
    this._cameraState = 'static';
  }

  /**
   * Advances all tracks to time `t` (seconds) and associates the new detections.
   * `camera` is the camera motion since the previous update (maps previous-frame
   * pixels to this frame), `null` if it could not be measured, or omitted for a
   * camera known to be static. Returns snapshots of all confirmed (and briefly
   * lost) tracks.
   */
  update(
    detections: Detection[],
    t: number,
    frameWidth: number,
    frameHeight: number,
    camera?: Homography | null,
  ): TrackSnapshot[] {
    const dt = this.lastT === null ? 0 : Math.max(0, t - this.lastT);
    const first = this.lastT === null;
    this.lastT = t;
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;
    const { highThreshold, lowThreshold } = this.opts;

    if (!first) this.applyCameraMotion(camera, t, dt);
    for (const tr of this.tracks) tr.kf.predict(dt);

    const high = detections.filter((d) => d.score >= highThreshold);
    const low = detections.filter((d) => d.score >= lowThreshold && d.score < highThreshold);

    // Stage 1: every track vs high-confidence detections (IoU, with a motion-aware
    // distance fallback so fast objects at low frame rates are not lost).
    const unmatchedTracks = new Set(this.tracks.map((_, i) => i));
    const unmatchedHigh = new Set(high.map((_, i) => i));
    const stage1 = assign(
      this.tracks.map((tr) => high.map((d) => this.cost(tr, d, true))),
      0.99,
    );
    for (const [ti, di] of stage1) {
      this.applyMatch(this.tracks[ti], high[di], t, frameWidth, frameHeight);
      unmatchedTracks.delete(ti);
      unmatchedHigh.delete(di);
    }

    // Stage 2: remaining active tracks vs low-confidence detections (IoU only).
    const remaining = [...unmatchedTracks].filter((i) => this.tracks[i].state !== 'lost');
    const stage2 = assign(
      remaining.map((i) => low.map((d) => this.cost(this.tracks[i], d, false))),
      0.7,
    );
    for (const [ri, di] of stage2) {
      const ti = remaining[ri];
      this.applyMatch(this.tracks[ti], low[di], t, frameWidth, frameHeight);
      unmatchedTracks.delete(ti);
    }

    // Unmatched tracks: coast, then expire.
    const dead = new Set<Track>();
    for (const ti of unmatchedTracks) {
      const tr = this.tracks[ti];
      tr.misses++;
      if (tr.state === 'tentative') {
        if (tr.misses > 1) dead.add(tr);
        continue;
      }
      tr.state = 'lost';
      const b = tr.kf.box;
      const outside = b.cx < 0 || b.cy < 0 || b.cx > frameWidth || b.cy > frameHeight;
      if (outside || t - tr.lastSeen > this.opts.maxLostSec) dead.add(tr);
    }
    this.tracks = this.tracks.filter((tr) => !dead.has(tr));

    // New tracks from unmatched high-confidence detections — unless the detection is
    // really a second box (often with another class) on an object we already track.
    for (const di of unmatchedHigh) {
      const d = high[di];
      if (this.tracks.some((tr) => iou(toDet(tr.kf.box), d) > DUPLICATE_IOU)) continue;
      const tr = new Track(this.nextId++, d, t);
      tr.history.push(this.historyPoint(tr, d, t, frameWidth, frameHeight));
      tr.vote(d.classId, d.score);
      // Very confident detections are shown immediately.
      if (this.opts.minHits <= 1 || d.score >= Math.min(0.9, highThreshold + 0.3)) this.confirm(tr);
      this.tracks.push(tr);
    }

    return this.snapshots(t);
  }

  private applyCameraMotion(camera: Homography | null | undefined, t: number, dt: number): void {
    let motion: Homography;
    if (camera) {
      motion = camera;
      this.lastCamera = { transform: camera, dt, t };
      const moved = cornerMotion(camera, this.frameWidth, this.frameHeight);
      this._cameraState = moved > Math.max(1, 0.002 * this.frameWidth) ? 'moving' : 'static';
    } else if (camera === null && this.lastCamera && t - this.lastCamera.t <= MAX_CAMERA_GAP_SEC && dt > 0) {
      // Background briefly unmeasurable (motion blur, a featureless patch): assume the
      // camera keeps moving as it just did.
      motion = scaleMotionH(this.lastCamera.transform, dt / Math.max(1e-3, this.lastCamera.dt));
      this._cameraState = 'moving';
    } else {
      motion = H_IDENTITY;
      this._cameraState = camera === null ? 'unknown' : 'static';
      if (camera === undefined) this.lastCamera = null;
    }
    if (motion === H_IDENTITY) return;
    this.pose = composeH(this.pose, invertH(motion));
    for (const tr of this.tracks) tr.kf.warp(motion);
  }

  private confirm(tr: Track): void {
    if (tr.state === 'tentative') this._totalConfirmed++;
    tr.state = 'confirmed';
  }

  private cost(tr: Track, d: Detection, allowDistance: boolean): number {
    if (!sameClassGroup(tr.classId, d.classId)) return INFEASIBLE;
    const classPenalty = tr.classId === d.classId ? 0 : 0.02;
    const pred = tr.kf.box;
    const overlap = iou(toDet(pred), d);
    if (overlap >= 0.1) return 1 - overlap + classPenalty;
    if (!allowDistance) return INFEASIBLE;
    const db = toBox(d);
    // Size must be plausible.
    const sizeRatio = Math.max(db.w, db.h) / Math.max(pred.w, pred.h);
    if (sizeRatio > 2.5 || sizeRatio < 0.4) return INFEASIBLE;
    const dist = Math.hypot(db.cx - pred.cx, db.cy - pred.cy);
    const diag = Math.hypot(pred.w, pred.h);
    const posStd = Math.hypot(tr.kf.cx.positionStd, tr.kf.cy.positionStd);
    const gate = 0.75 * diag + 3 * posStd;
    if (dist >= gate) return INFEASIBLE;
    return 0.9 + 0.08 * (dist / gate) + classPenalty / 2;
  }

  private applyMatch(tr: Track, d: Detection, t: number, fw: number, fh: number): void {
    const box = toBox(d);
    tr.kf.update(box);
    tr.hits++;
    tr.misses = 0;
    tr.lastSeen = t;
    tr.score = d.score;
    tr.vote(d.classId, d.score);
    tr.history.push(this.historyPoint(tr, d, t, fw, fh));
    if (tr.history.length > MAX_HISTORY) tr.history.shift();
    if (tr.state === 'lost') tr.state = 'confirmed';
    else if (tr.state === 'tentative' && tr.hits >= this.opts.minHits) this.confirm(tr);
  }

  /** Camera focal length in pixels (from the assumed zoom). */
  private get focalPx(): number {
    return focalLengthPx(this.frameWidth, this.frameHeight, zoomToDiagonalFov(this.opts.zoom));
  }

  /** Rotation from the current camera to the reference camera, or null if there is none. */
  private poseRotation(): number[] | null {
    if (this.pose === H_IDENTITY) return null;
    return rotationFromH(this.pose, this.focalPx, this.frameWidth / 2, this.frameHeight / 2);
  }

  /**
   * History positions are kept in the scene-fixed reference frame, so camera motion
   * cancels out; the box itself is kept as seen, with the camera orientation it was
   * seen from.
   */
  private historyPoint(tr: Track, d: Detection, t: number, fw: number, fh: number): HistoryPoint {
    const margin = 2;
    const b = toBox(d);
    const [cx, cy] = applyH(this.pose, b.cx, b.cy);
    const [sx, sy] = applyH(this.pose, tr.kf.cx.p, tr.kf.cy.p);
    const rot = this.poseRotation();
    return {
      t,
      cx,
      cy,
      w: b.w,
      h: b.h,
      sx,
      sy,
      clipped: d.x1 <= margin || d.y1 <= margin || d.x2 >= fw - margin || d.y2 >= fh - margin,
      ...(rot ? { ix: b.cx, iy: b.cy, rot } : {}),
    };
  }

  /** The recent history seen through the current camera (current-frame pixels). */
  private historyInView(tr: Track, t: number, maxAgeSec: number): HistoryPoint[] {
    const toView = invertH(this.pose);
    const now = this.poseRotation();
    const out: HistoryPoint[] = [];
    let i = tr.history.length - 1;
    while (i > 0 && t - tr.history[i - 1].t <= maxAgeSec) i--;
    for (; i < tr.history.length; i++) {
      const p = tr.history[i];
      const [cx, cy] = applyH(toView, p.cx, p.cy);
      const [sx, sy] = applyH(toView, p.sx ?? p.cx, p.sy ?? p.cy);
      const q: HistoryPoint = { t: p.t, cx, cy, w: p.w, h: p.h, sx, sy, clipped: p.clipped };
      if (p.rot || now) {
        // Seen from another camera orientation: keep where it was in that frame and
        // the rotation from that camera to the current one.
        const id = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        q.ix = p.ix ?? p.cx;
        q.iy = p.iy ?? p.cy;
        q.rot = transposeMul(now ?? id, p.rot ?? id);
      }
      out.push(q);
    }
    return out;
  }

  private snapshots(t: number): TrackSnapshot[] {
    const out: TrackSnapshot[] = [];
    for (const tr of this.tracks) {
      if (tr.state === 'tentative') continue;
      out.push(this.snapshot(tr, t));
    }
    return out;
  }

  private snapshot(tr: Track, t: number): TrackSnapshot {
    const { velocityWindowSec, turnWindowSec, trailSec } = this.opts;
    const b = tr.kf.box;
    const fullHistory = this.historyInView(tr, t, Math.max(trailSec, 2.5 * Math.max(velocityWindowSec, 2, turnWindowSec)));
    // A box cut off by the frame edge moves and grows as the object slides into view,
    // which is not the object's motion: measure from whole boxes when there are enough.
    const whole = fullHistory.filter((p) => !p.clipped && t - p.t <= 2.5 * velocityWindowSec);
    const partial = whole.length < 4;
    const history = partial ? fullHistory : fullHistory.filter((p) => !p.clipped);
    if (tr.state !== 'lost') {
      const fit = fitVelocity(history, velocityWindowSec);
      tr.lastVelocity = fit ?? { vx: tr.kf.cx.v, vy: tr.kf.cy.v };
      const span = history.length ? history[history.length - 1].t - history[0].t : 0;
      tr.measuring = fit === null || history.length < 4 || span < MIN_SPEED_SPAN_SEC || (partial && tr.speedEma === null);
      tr.lastTurnRate = 0.5 * tr.lastTurnRate + 0.5 * estimateTurnRate(history, turnWindowSec);
    } else if (this.lastCamera && this._cameraState === 'moving') {
      // Keep the coasting velocity in the current camera's orientation.
      const [vx, vy] = applyHLinear(this.lastCamera.transform, b.cx, b.cy, tr.lastVelocity.vx, tr.lastVelocity.vy);
      tr.lastVelocity = { vx, vy };
    }
    const { vx, vy } = tr.lastVelocity;
    const speedPx = Math.hypot(vx, vy);
    const size = Math.max(b.w, b.h);
    let measuring = tr.measuring;
    let speedKmh: number | null = null;
    let approach: TrackSnapshot['approach'] = null;
    let depthMoving = false;
    /** Showing the previous reading instead of a new measurement. */
    let holding = partial;
    const ref = this.opts.referenceSizes[tr.classId] ?? REFERENCE_SIZE[tr.classId];
    if (ref && !measuring) {
      // Only partly in view: hold the last reading rather than measure the cut-off box.
      const stale = !tr.lastEstimate || t - tr.lastEstimate.t >= SPEED_UPDATE_SEC - 1e-6 || t < tr.lastEstimate.t;
      if (stale && !partial) {
        const fresh = estimateSpeed({
          history,
          frameWidth: this.frameWidth,
          frameHeight: this.frameHeight,
          ref,
          windowSec: velocityWindowSec,
          focalPx: this.focalPx,
        });
        tr.lastEstimate = { t, est: fresh };
      }
      const est = tr.lastEstimate?.est ?? null;
      if (est && !est.settled) {
        // Undecided whether it moves towards / away from the camera: keep showing the
        // last reading, or "measuring" if there is none yet.
        if (tr.speedEma === null) measuring = true;
        else speedKmh = tr.speedEma;
        holding = true;
      } else if (est) {
        speedKmh = est.metresPerSecond * MS_TO_KMH;
        if (est.depthComponent <= -0.5) approach = 'approaching';
        else if (est.depthComponent >= 0.5) approach = 'receding';
        depthMoving = approach !== null && speedKmh > 3;
        if (speedKmh > (MAX_PLAUSIBLE_KMH[tr.classId] ?? Infinity)) {
          speedKmh = null;
          approach = null;
          measuring = true;
        }
      }
    }
    // Head-on objects barely move on screen but still loom, so they are not stationary.
    let stationary = !measuring && !holding && !depthMoving && speedPx < Math.max(4, 0.15 * size);
    if (stationary) {
      if (speedKmh !== null) speedKmh = 0;
      approach = null;
    }
    if (speedKmh !== null && tr.state !== 'lost' && tr.speedEmaT !== t && !holding) {
      // A median over the last second rejects one-off glitches (a bad box, a
      // mis-measured camera move); the EMA then steadies the readout.
      tr.speedSamples.push({ t, v: speedKmh });
      while (tr.speedSamples.length && t - tr.speedSamples[0].t > SPEED_MEDIAN_SEC) tr.speedSamples.shift();
      const filtered = median(tr.speedSamples.map((p) => p.v));
      const alpha = tr.speedEma === null ? 1 : 1 - Math.exp(-Math.max(0, t - tr.speedEmaT) / SPEED_SMOOTHING_SEC);
      tr.speedEma = tr.speedEma === null ? filtered : tr.speedEma + alpha * (filtered - tr.speedEma);
      tr.speedEmaT = t;
      speedKmh = tr.speedEma;
    } else if (speedKmh !== null) {
      speedKmh = tr.speedEma ?? speedKmh;
    }
    // A reading that rounds to nothing is "stationary" (no meaningful heading).
    if (!holding && speedKmh !== null && speedKmh < 1) stationary = true;
    if (stationary && speedKmh !== null) speedKmh = 0;
    if (speedKmh !== null && speedKmh < 3) approach = null;
    const headingDeg = stationary || measuring ? null : headingDegrees(vx, vy);

    // On-screen motion = own motion + the camera's (for drawing between detections).
    let screenVx = vx;
    let screenVy = vy;
    if (this.lastCamera && this._cameraState === 'moving' && this.lastCamera.dt > 0) {
      const [mx, my] = applyH(this.lastCamera.transform, b.cx, b.cy);
      screenVx += (mx - b.cx) / this.lastCamera.dt;
      screenVy += (my - b.cy) / this.lastCamera.dt;
    }

    const trail: TrackSnapshot['trail'] = [];
    for (const p of fullHistory) {
      if (t - p.t <= trailSec) trail.push({ t: p.t, x: p.sx ?? p.cx, y: p.sy ?? p.cy });
    }

    return {
      id: tr.id,
      classId: tr.classId,
      label: className(tr.classId),
      score: tr.score,
      state: tr.state,
      t,
      firstSeen: tr.firstSeen,
      cx: b.cx,
      cy: b.cy,
      w: b.w,
      h: b.h,
      vx,
      vy,
      screenVx,
      screenVy,
      vw: tr.kf.w.v,
      vh: tr.kf.h.v,
      speedPx,
      speedKmh,
      headingDeg,
      compass: headingDeg === null ? null : compassLabel(headingDeg),
      stationary,
      measuring,
      approach,
      turnRate: stationary || measuring ? 0 : tr.lastTurnRate,
      trail,
    };
  }
}

