import { CLASS_ID, MAX_PLAUSIBLE_KMH, REFERENCE_SIZE, className } from '../detection/classes';
import { iou } from '../detection/postprocess';
import type { Detection } from '../detection/types';
import { assign } from './assignment';
import { BoxKalman, type Box } from './kalman';
import {
  MS_TO_KMH,
  compassLabel,
  estimateSpeed,
  estimateTurnRate,
  fitVelocity,
  headingDegrees,
} from './motion';
import type { HistoryPoint, TrackSnapshot, TrackState } from './types';

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
}

export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
  highThreshold: 0.4,
  lowThreshold: 0.1,
  minHits: 2,
  maxLostSec: 1.0,
  velocityWindowSec: 0.6,
  turnWindowSec: 1.2,
  trailSec: 3,
};

/** Classes the detector often confuses with each other; matching across them is allowed. */
const CLASS_GROUPS: number[][] = [
  [CLASS_ID.car, CLASS_ID.truck, CLASS_ID.bus, CLASS_ID.train],
  [CLASS_ID.bicycle, CLASS_ID.motorcycle],
  [CLASS_ID.airplane, CLASS_ID.bird, CLASS_ID.kite],
  [CLASS_ID.boat, CLASS_ID.surfboard],
];
const GROUP_OF = new Map<number, number>();
CLASS_GROUPS.forEach((g, i) => g.forEach((c) => GROUP_OF.set(c, i)));

function sameGroup(a: number, b: number): boolean {
  if (a === b) return true;
  const ga = GROUP_OF.get(a);
  return ga !== undefined && ga === GROUP_OF.get(b);
}

const MAX_HISTORY = 400;
const SPEED_SMOOTHING_SEC = 0.5;
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
  /** Exponentially smoothed speed for display, km/h. */
  speedEma: number | null = null;
  speedEmaT = 0;
  lastTurnRate = 0;

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

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
    this.lastT = null;
    this._totalConfirmed = 0;
  }

  /**
   * Advances all tracks to time `t` (seconds) and associates the new detections.
   * Returns snapshots of all confirmed (and briefly lost) tracks.
   */
  update(detections: Detection[], t: number, frameWidth: number, frameHeight: number): TrackSnapshot[] {
    const dt = this.lastT === null ? 0 : Math.max(0, t - this.lastT);
    this.lastT = t;
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;
    const { highThreshold, lowThreshold } = this.opts;

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

  private confirm(tr: Track): void {
    if (tr.state === 'tentative') this._totalConfirmed++;
    tr.state = 'confirmed';
  }

  private cost(tr: Track, d: Detection, allowDistance: boolean): number {
    if (!sameGroup(tr.classId, d.classId)) return INFEASIBLE;
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

  private historyPoint(tr: Track, d: Detection, t: number, fw: number, fh: number): HistoryPoint {
    const margin = 2;
    return {
      t,
      ...toBox(d),
      sx: tr.kf.cx.p,
      sy: tr.kf.cy.p,
      clipped: d.x1 <= margin || d.y1 <= margin || d.x2 >= fw - margin || d.y2 >= fh - margin,
    };
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
    if (tr.state !== 'lost') {
      const fit = fitVelocity(tr.history, velocityWindowSec);
      tr.lastVelocity = fit ?? { vx: tr.kf.cx.v, vy: tr.kf.cy.v };
      tr.measuring = fit === null;
      tr.lastTurnRate = 0.5 * tr.lastTurnRate + 0.5 * estimateTurnRate(tr.history, turnWindowSec);
    }
    const { vx, vy } = tr.lastVelocity;
    const speedPx = Math.hypot(vx, vy);
    const size = Math.max(b.w, b.h);
    let measuring = tr.measuring;
    let speedKmh: number | null = null;
    let approach: TrackSnapshot['approach'] = null;
    let depthMoving = false;
    const ref = REFERENCE_SIZE[tr.classId];
    if (ref && !measuring) {
      const est = estimateSpeed({
        history: tr.history,
        vx,
        vy,
        cx: b.cx,
        cy: b.cy,
        frameWidth: this.frameWidth,
        frameHeight: this.frameHeight,
        ref,
        windowSec: velocityWindowSec,
      });
      if (est) {
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
    const stationary = !measuring && !depthMoving && speedPx < Math.max(4, 0.15 * size);
    if (stationary) {
      if (speedKmh !== null) speedKmh = 0;
      approach = null;
    }
    if (speedKmh !== null && tr.state !== 'lost') {
      // Time-based EMA (τ = 0.5 s) so the readout is steady at any frame rate.
      const alpha = tr.speedEma === null ? 1 : 1 - Math.exp(-Math.max(0, t - tr.speedEmaT) / SPEED_SMOOTHING_SEC);
      tr.speedEma = tr.speedEma === null ? speedKmh : tr.speedEma + alpha * (speedKmh - tr.speedEma);
      tr.speedEmaT = t;
      speedKmh = tr.speedEma;
    } else if (speedKmh !== null) {
      speedKmh = tr.speedEma ?? speedKmh;
    }
    const headingDeg = stationary || measuring ? null : headingDegrees(vx, vy);

    const trail: TrackSnapshot['trail'] = [];
    for (let i = tr.history.length - 1; i >= 0 && t - tr.history[i].t <= trailSec; i--) {
      const p = tr.history[i];
      trail.push({ t: p.t, x: p.sx ?? p.cx, y: p.sy ?? p.cy });
    }
    trail.reverse();

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
