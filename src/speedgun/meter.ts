/**
 * The speed gun's measuring logic (no DOM, so it can be unit-tested with simulated
 * scenes). Each video frame it receives:
 *   - the time of the frame (frame index / FPS),
 *   - how each followed vehicle moved since the previous frame (sub-pixel optical flow),
 *   - how the camera itself moved (so a panning camera is not mistaken for speed),
 *   - every few frames, fresh detector boxes (to find new vehicles and re-anchor boxes),
 * and turns the pixel displacements into metres with the user's calibration line.
 *
 * Two calibration modes, because one metres-per-pixel ratio is only true at one
 * distance from the camera:
 *
 *  GROUND ("speed trap") — the line is drawn on the road (e.g. along lane markings).
 *    The ratio is exact only where the line is, so — like a real speed camera — each
 *    vehicle is measured while its ground-contact point (bottom of its box) is inside
 *    the zone the line spans. When the camera looks along the road, the image row of a
 *    point on the road depends only on how far down the road it is, so the line's
 *    *vertical* extent gives "metres per vertical pixel" for every lane at that distance
 *    (a line drawn across the view uses the horizontal extent instead). A vehicle that
 *    crosses the whole zone gets the exact trap reading: line length ÷ crossing time.
 *
 *  OBJECT — the line is drawn along the vehicle itself (car length, plane fuselage).
 *    The ratio then follows the vehicle: when it appears 2× bigger (closer, or the
 *    camera zooms in) one pixel is half as many metres. Its displacement is measured
 *    relative to the background, so a camera panning to follow it does not hide its speed.
 */

import { iou } from '../detection/postprocess';
import type { Detection } from '../detection/types';
import { assign } from '../tracking/assignment';
import { applyAffine, moveRect, type Affine } from '../vision/regionTracker';
import { applyH, applySim, type Homography, type Similarity } from '../vision/transforms';
import {
  fitSlope,
  fitSpeed2D,
  lastWindow,
  lineLengthPx,
  metresPerPixel,
  toKmh,
  type Point,
  type Sample,
} from './speedMath';

export type CalibrationMode = 'ground' | 'object';

export interface Calibration {
  /** Line end points on the calibration frame, video pixels. */
  a: Point;
  b: Point;
  /** Real length of the line, metres. */
  metres: number;
  mode: CalibrationMode;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** The measuring zone of a ground calibration. */
export interface Trap {
  /** Which pixel component measures distance along the road. */
  axis: 'x' | 'y';
  lo: number;
  hi: number;
  /** Metres per pixel along `axis` (line length ÷ its extent on that axis). */
  mPerAxisPx: number;
  metres: number;
}

export function makeTrap(cal: Calibration): Trap {
  const dx = Math.abs(cal.b.x - cal.a.x);
  const dy = Math.abs(cal.b.y - cal.a.y);
  const axis = dy >= dx ? 'y' : 'x';
  const lo = Math.min(cal.a[axis], cal.b[axis]);
  const hi = Math.max(cal.a[axis], cal.b[axis]);
  return { axis, lo, hi, mPerAxisPx: metresPerPixel(cal.metres, hi - lo), metres: cal.metres };
}

/** Where a vehicle touches the ground: the middle of the bottom edge of its box. */
export function groundPoint(b: Box): Point {
  return { x: (b.x1 + b.x2) / 2, y: b.y2 };
}

/** True if the box touches the frame border, i.e. part of the vehicle may be cut off. */
export function touchesEdge(b: Box, width: number, height: number, margin = 4): boolean {
  return b.x1 <= margin || b.y1 <= margin || b.x2 >= width - margin || b.y2 >= height - margin;
}

/** Intersection area divided by the smaller box's area (1 = one box inside the other). */
export function overlapOfSmaller(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const small = Math.min((a.x2 - a.x1) * (a.y2 - a.y1), (b.x2 - b.x1) * (b.y2 - b.y1));
  return small > 0 ? (ix * iy) / small : 0;
}

export function centre(b: Box): Point {
  return { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 };
}

export type TrackStatus = 'approaching' | 'measuring' | 'measured' | 'missed' | 'tracking';

export interface GunTrack {
  id: number;
  classId: number;
  box: Box;
  /** Own motion of the box centre in the last frame, px (start guess for optical flow). */
  lastShift: Point;
  /** Consecutive frames without optical-flow tracking. */
  misses: number;
  /** Cumulative distance travelled (metres) at each measured frame. */
  samples: Array<{ t: number; x: number; y: number }>;
  status: TrackStatus;
  /** Live speed, km/h (null until enough samples). */
  currentKmh: number | null;
  /** Final reading after crossing the trap zone, km/h. */
  finalKmh: number | null;
  /** True when `finalKmh` came from a full zone crossing (line length ÷ crossing time). */
  fullCrossing: boolean;
  trail: Point[];
  /** Frames since the vehicle left the zone (finished tracks are retired after a while). */
  sinceExit: number;
  /**
   * The box's ground edge is the vehicle's own (confirmed by a detection that is not cut
   * by the frame border). Optical flow alone cannot fix a box that started out clipped.
   */
  groundOk: boolean;
  groundOkPrev: boolean;
  // Ground-mode zone bookkeeping.
  enterT: number | null;
  enterBound: 'lo' | 'hi' | null;
  zoneSamples: Sample[];
  // Object-mode scale: how much bigger the object looks now than on the calibration frame.
  scale: number;
  /**
   * Object mode: the calibration line's two ends, carried along on the vehicle frame by
   * frame (they stay on the nose and tail). Its current length is what the known real
   * length spans now — it grows as the vehicle comes closer, the camera zooms in, or
   * the vehicle turns side-on.
   */
  line: { a: Point; b: Point } | null;
}

export interface FrameInput {
  t: number;
  /** Camera motion previous → this frame; undefined = no previous frame; null = unmeasurable. */
  camera: Homography | null | undefined;
  /** Optical-flow motion of each track (by id) previous → this frame; null = lost. */
  motions: ReadonlyMap<number, Similarity | null>;
  /** The same motions as affine maps, when available (they also follow turning objects). */
  affines?: ReadonlyMap<number, Affine | null>;
  /** Object mode: affine motion of the band around the calibration line, when available. */
  lineAffines?: ReadonlyMap<number, Affine | null>;

  /** Detector boxes, on the frames the detector ran. */
  detections?: readonly Detection[];
}

export interface MeterOptions {
  /** Window for the live speed, seconds. */
  windowSec: number;
  /** Frames a track may go without optical flow before it is dropped (ground mode). */
  maxMisses: number;
}

const DEFAULTS: MeterOptions = { windowSec: 0.5, maxMisses: 8 };

export class SpeedMeter {
  readonly tracks: GunTrack[] = [];
  readonly trap: Trap | null;
  /** Fastest reading so far, km/h. */
  maxKmh: number | null = null;
  /** Most recent reading (live in-zone or just-finished), km/h. */
  currentKmh: number | null = null;
  currentId: number | null = null;
  private nextId = 1;
  private lastT: number | null = null;
  private lastCamera: { H: Homography; t: number } | null = null;
  private readonly calPx: number;
  private readonly dir: Point;
  private readonly opts: MeterOptions;

  constructor(
    readonly cal: Calibration,
    private frameWidth: number,
    private frameHeight: number,
    opts: Partial<MeterOptions> = {},
  ) {
    this.opts = { ...DEFAULTS, ...opts };
    this.trap = cal.mode === 'ground' ? makeTrap(cal) : null;
    this.calPx = lineLengthPx(cal.a, cal.b);
    this.dir = { x: (cal.b.x - cal.a.x) / this.calPx, y: (cal.b.y - cal.a.y) / this.calPx };
  }

  /** The calibration ratio from the spec: metres per pixel along the drawn line. */
  get metresPerPixel(): number {
    return metresPerPixel(this.cal.metres, this.calPx);
  }

  /**
   * Object mode: start following the vehicle the line was drawn on. Picks the detection
   * containing the line's middle; without one, a box around the line itself.
   */
  initObject(detections: readonly Detection[]): GunTrack {
    const mid = { x: (this.cal.a.x + this.cal.b.x) / 2, y: (this.cal.a.y + this.cal.b.y) / 2 };
    const hits = detections.filter((d) => mid.x >= d.x1 && mid.x <= d.x2 && mid.y >= d.y1 && mid.y <= d.y2);
    // The detection that best matches the line's length is the vehicle that was measured.
    hits.sort((p, q) => Math.abs(this.extent(p) - this.calPx) - Math.abs(this.extent(q) - this.calPx));
    const det = hits[0];
    let box: Box;
    if (det) {
      box = { x1: det.x1, y1: det.y1, x2: det.x2, y2: det.y2 };
    } else {
      const pad = 0.15 * this.calPx;
      box = {
        x1: Math.min(this.cal.a.x, this.cal.b.x) - pad,
        y1: Math.min(this.cal.a.y, this.cal.b.y) - pad,
        x2: Math.max(this.cal.a.x, this.cal.b.x) + pad,
        y2: Math.max(this.cal.a.y, this.cal.b.y) + pad,
      };
    }
    const tr = this.newTrack(box, det?.classId ?? -1);
    tr.line = { a: { ...this.cal.a }, b: { ...this.cal.b } };
    tr.status = 'tracking';
    return tr;
  }

  /**
   * Road mode: a vehicle whose ground point is not confirmed yet (it is still cut by the
   * frame border) gets a fresh detection on the next frame, so the moment its true
   * bottom edge appears — often just before it enters the zone — is not missed.
   */
  needsDetection(): boolean {
    return this.trap !== null && this.tracks.some((t) => !t.groundOk && (t.status === 'approaching' || t.status === 'measuring'));
  }

  /** Boxes (and, in object mode, the calibration line) to follow into the next frame. */
  targets() {
    return this.tracks.map((t) => ({
      id: t.id,
      box: { ...t.box },
      guess: { ...t.lastShift },
      ...(t.line ? { line: { a: { ...t.line.a }, b: { ...t.line.b } } } : {}),
    }));
  }

  update(input: FrameInput): void {
    const { t } = input;
    const dt = this.lastT === null ? 0 : t - this.lastT;
    this.lastT = t;
    // Camera motion: unknown for a moment (e.g. a blurred frame) → assume it kept moving
    // as before, as cameras pan smoothly; never assume that for long.
    let H: Homography | null = null;
    if (input.camera) {
      H = input.camera;
      this.lastCamera = { H, t };
    } else if (input.camera === null && this.lastCamera && t - this.lastCamera.t < 0.5) {
      H = this.lastCamera.H;
    }
    const comparable = input.camera !== undefined && dt > 0 && H !== null;
    /** Vehicle mode: both ends of the line must be in the picture for its length to mean anything. */
    const lineVisible = (l: { a: Point; b: Point } | null) =>
      !l || [l.a, l.b].every((p) => p.x >= 2 && p.y >= 2 && p.x <= this.frameWidth - 2 && p.y <= this.frameHeight - 2);

    for (const tr of this.tracks) {
      const motion = input.motions.get(tr.id) ?? null;
      const prevBox = tr.box;
      tr.groundOkPrev = tr.groundOk;
      const prevLine = tr.line ? { a: { ...tr.line.a }, b: { ...tr.line.b } } : null;
      if (motion) {
        tr.box = moveRect(prevBox, motion);
        tr.misses = 0;
        if (tr.line) {
          const aff = input.lineAffines?.get(tr.id) ?? input.affines?.get(tr.id) ?? null;
          const map = (p: Point): Point => {
            const [x, y] = aff ? applyAffine(aff, p.x, p.y) : applySim(motion, p.x, p.y);
            return { x, y };
          };
          tr.line = { a: map(tr.line.a), b: map(tr.line.b) };
          tr.scale = lineLengthPx(tr.line.a, tr.line.b) / this.calPx;
        }
      } else {
        // Not followed this frame: coast with the camera and the last own motion.
        const c = centre(prevBox);
        const [cx, cy] = H ? applyH(H, c.x, c.y) : [c.x, c.y];
        const sx = cx - c.x + tr.lastShift.x;
        const sy = cy - c.y + tr.lastShift.y;
        tr.box = { x1: prevBox.x1 + sx, y1: prevBox.y1 + sy, x2: prevBox.x2 + sx, y2: prevBox.y2 + sy };
        if (tr.line) {
          tr.line = {
            a: { x: tr.line.a.x + sx, y: tr.line.a.y + sy },
            b: { x: tr.line.b.x + sx, y: tr.line.b.y + sy },
          };
        }
        tr.misses++;
      }
      // No reading while the length reference is partly out of the picture: hold the last one.
      const usable = comparable && motion && lineVisible(prevLine) && lineVisible(tr.line);
      if (usable) this.measure(tr, prevBox, H!, t, dt, prevLine);
      else if (tr.line) tr.samples.length = 0; // restart the averaging window afterwards
      else if (this.cal.mode === 'ground' && dt > 0) this.zoneBookkeeping(tr, prevBox, t, dt, false);
      const c = centre(tr.box);
      tr.trail.push(this.cal.mode === 'ground' ? groundPoint(tr.box) : c);
      if (tr.trail.length > 90) tr.trail.shift();
    }

    if (input.detections) this.associate(input.detections);
    this.prune();
    this.updateSummary();
  }

  // ------------------------------------------------------------- measuring

  /** One frame of "distance over time" for one vehicle. */
  private measure(
    tr: GunTrack,
    prevBox: Box,
    H: Homography,
    t: number,
    dt: number,
    prevLine: { a: Point; b: Point } | null,
  ) {
    const ground = this.cal.mode === 'ground';
    const mid = (l: { a: Point; b: Point }) => ({ x: (l.a.x + l.b.x) / 2, y: (l.a.y + l.b.y) / 2 });
    // Reference point: ground contact (trap) or the middle of the line on the vehicle.
    const pPrev = ground ? groundPoint(prevBox) : prevLine ? mid(prevLine) : centre(prevBox);
    const pNow = ground ? groundPoint(tr.box) : tr.line ? mid(tr.line) : centre(tr.box);
    // Where the point would be if the vehicle had stood still and only the camera moved.
    const [sx, sy] = applyH(H, pPrev.x, pPrev.y);
    // The vehicle's own displacement this frame, pixels.
    const dx = pNow.x - sx;
    const dy = pNow.y - sy;
    const cPrev = centre(prevBox);
    const cNow = centre(tr.box);
    const [ccx, ccy] = applyH(H, cPrev.x, cPrev.y);
    tr.lastShift = { x: cNow.x - ccx, y: cNow.y - ccy };

    const last = tr.samples[tr.samples.length - 1] ?? { t, x: 0, y: 0 };
    if (ground) {
      // Distance along the road = pixels along the trap axis × metres per axis pixel.
      const trap = this.trap!;
      const d = (trap.axis === 'y' ? dy : dx) * trap.mPerAxisPx;
      tr.samples.push({ t, x: 0, y: last.y + d });
      this.zoneBookkeeping(tr, prevBox, t, dt, true);
    } else {
      // Metres per pixel *now*: the known length divided by the line's current length on
      // the vehicle (averaged over the frame step, as the vehicle grows or shrinks).
      const lenNow = tr.line ? lineLengthPx(tr.line.a, tr.line.b) : this.calPx * tr.scale;
      const lenPrev = prevLine ? lineLengthPx(prevLine.a, prevLine.b) : lenNow;
      const mpp = this.cal.metres / ((lenNow + lenPrev) / 2);
      tr.samples.push({ t, x: last.x + dx * mpp, y: last.y + dy * mpp });
      if (tr.samples.length > 3000) tr.samples.shift();
      const w = lastWindow(tr.samples, this.opts.windowSec);
      const v = w.length >= 4 && w[w.length - 1].t - w[0].t >= 0.2 ? fitSpeed2D(tr.samples, this.opts.windowSec) : null;
      tr.currentKmh = v === null ? null : toKmh(v);
    }
  }

  /** Ground mode: is the vehicle in the zone, did it just enter / leave it? */
  private zoneBookkeeping(tr: GunTrack, prevBox: Box, t: number, dt: number, measured: boolean) {
    const trap = this.trap!;
    const a0 = groundPoint(prevBox)[trap.axis];
    const a1 = groundPoint(tr.box)[trap.axis];
    const inside = (a: number) => a >= trap.lo && a <= trap.hi;
    if (tr.status === 'measured' || tr.status === 'missed') {
      tr.sinceExit++;
      return;
    }
    // While the box is cut by the frame border on the side we measure (its bottom edge for
    // a road running up the picture), that edge is the border, not the vehicle's ground
    // point, so it cannot start or end a measurement. A car cut off on its left or right
    // still has a true bottom edge.
    const clipped = !tr.groundOk || !tr.groundOkPrev || this.groundClipped(prevBox) || this.groundClipped(tr.box);
    if (clipped) {
      if (tr.status === 'measuring' && measured && inside(a1)) {
        tr.zoneSamples.push({ t, d: tr.samples[tr.samples.length - 1].y });
      }
      return;
    }
    // Exact moment the ground point crossed a zone boundary (linear between frames).
    const crossT = (bound: number) => t - dt + ((bound - a0) / (a1 - a0)) * dt;

    if (!inside(a0) && inside(a1) && tr.status === 'approaching') {
      const bound = Math.abs(a0 - trap.lo) < Math.abs(a0 - trap.hi) ? 'lo' : 'hi';
      tr.enterBound = bound;
      tr.enterT = crossT(trap[bound]);
      tr.status = 'measuring';
    } else if (inside(a1) && tr.status === 'approaching') {
      // First seen already inside the zone: measured, but not a full crossing.
      tr.status = 'measuring';
    }
    if (tr.status === 'measuring' && inside(a1) && measured) {
      tr.zoneSamples.push({ t, d: tr.samples[tr.samples.length - 1].y });
      // Live reading: a vehicle's speed hardly changes within the zone, so fit all of
      // its zone samples (up to 1.5 s) — far steadier than the last few frames alone.
      const w = lastWindow(tr.zoneSamples, 1.5);
      const f = w.length >= 5 && w[w.length - 1].t - w[0].t >= 0.12 ? fitSlope(w) : null;
      tr.currentKmh = f ? Math.abs(toKmh(f.slope)) : tr.currentKmh;
    }
    if (tr.status === 'measuring' && inside(a0) && !inside(a1)) {
      // Left the zone. Full crossing → line length ÷ crossing time (the trap reading);
      // otherwise the fitted speed over the part of the zone it was seen in.
      const bound = a1 < trap.lo ? 'lo' : 'hi';
      if (tr.enterBound && bound !== tr.enterBound && tr.enterT !== null) {
        const exitT = crossT(trap[bound]);
        tr.finalKmh = toKmh(trap.metres / (exitT - tr.enterT));
        tr.fullCrossing = true;
      } else {
        tr.finalKmh = this.partialReading(tr);
      }
      tr.status = tr.finalKmh === null ? 'missed' : 'measured';
      tr.currentKmh = tr.finalKmh;
      if (tr.finalKmh !== null) {
        this.currentKmh = tr.finalKmh;
        this.currentId = tr.id;
      }
    }
  }

  /** The box's ground point is not trustworthy (cut by the border on the measured side). */
  private groundClipped(b: Box): boolean {
    const m = 4;
    if (this.trap!.axis === 'y') return b.y2 >= this.frameHeight - m;
    // A road running across the picture: both ends of the vehicle matter.
    return b.x1 <= m || b.x2 >= this.frameWidth - m || b.y2 >= this.frameHeight - m;
  }

  /** A partial reading is only trusted if the vehicle covered at least half the zone. */
  private partialReading(tr: GunTrack): number | null {
    const zs = tr.zoneSamples;
    if (zs.length < 5) return null;
    const covered = Math.abs(zs[zs.length - 1].d - zs[0].d);
    const f = fitSlope(zs);
    return f && covered >= 0.5 * this.trap!.metres ? Math.abs(toKmh(f.slope)) : null;
  }

  // ------------------------------------------------------- detections & tracks

  /** Box extent along the calibration line's direction (≈ the measured length), px. */
  private extent(b: Box): number {
    return Math.abs(this.dir.x) * (b.x2 - b.x1) + Math.abs(this.dir.y) * (b.y2 - b.y1);
  }

  private newTrack(box: Box, classId: number): GunTrack {
    const tr: GunTrack = {
      id: this.nextId++,
      classId,
      box,
      lastShift: { x: 0, y: 0 },
      misses: 0,
      samples: [],
      status: 'approaching',
      currentKmh: null,
      finalKmh: null,
      fullCrossing: false,
      trail: [],
      sinceExit: 0,
      groundOk: false,
      groundOkPrev: false,
      enterT: null,
      enterBound: null,
      zoneSamples: [],
      scale: 1,
      line: null,
    };
    this.tracks.push(tr);
    return tr;
  }

  private associate(dets: readonly Detection[]) {
    const cost = this.tracks.map((tr) => dets.map((d) => 1 - iou({ ...tr.box, score: 0, classId: 0 }, d)));
    const pairs = assign(cost, 0.7);
    const used = new Set<number>();
    for (const [ti, di] of pairs) {
      const tr = this.tracks[ti];
      const d = dets[di];
      used.add(di);
      // Pull the box towards the detection (optical flow drifts slowly; the detector does
      // not). Distances were already measured from the flow, so this does not add jitter.
      // Snap fully when the flow lost the vehicle or either box is cut by the frame border
      // (a clipped box's bottom edge is the border, not the vehicle).
      const W = this.frameWidth;
      const Hh = this.frameHeight;
      const k = tr.misses || touchesEdge(tr.box, W, Hh) || touchesEdge(d, W, Hh) ? 1 : 0.35;
      tr.box = {
        x1: tr.box.x1 + k * (d.x1 - tr.box.x1),
        y1: tr.box.y1 + k * (d.y1 - tr.box.y1),
        x2: tr.box.x2 + k * (d.x2 - tr.box.x2),
        y2: tr.box.y2 + k * (d.y2 - tr.box.y2),
      };
      if (tr.classId < 0) tr.classId = d.classId;
      if (this.trap) tr.groundOk = !this.groundClipped(d);
    }
    if (this.cal.mode !== 'ground') return;
    dets.forEach((d, i) => {
      if (used.has(i)) return;
      // Not a new vehicle if it mostly overlaps one we already follow (a growing box of a
      // car entering the picture can have a low IoU with the track yet be the same car).
      if (this.tracks.some((tr) => overlapOfSmaller(tr.box, d) > 0.5)) return;
      const tr = this.newTrack({ x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 }, d.classId);
      tr.groundOk = tr.groundOkPrev = !this.groundClipped(d);
    });
  }

  private prune() {
    const W = this.frameWidth;
    const H = this.frameHeight;
    // Two tracks on one vehicle: keep the older (it has the longer history).
    for (let i = this.tracks.length - 1; i >= 0; i--) {
      const young = this.tracks[i];
      if (young.status === 'measured' || this.cal.mode === 'object') continue;
      const dup = this.tracks.some((old) => old.id < young.id && overlapOfSmaller(old.box, young.box) > 0.7);
      if (dup) this.tracks.splice(i, 1);
    }
    for (let i = this.tracks.length - 1; i >= 0; i--) {
      const tr = this.tracks[i];
      const c = centre(tr.box);
      const outside = c.x < 0 || c.y < 0 || c.x > W || c.y > H;
      if (this.cal.mode === 'object') continue; // never give up on the calibrated vehicle
      const finished = (tr.status === 'measured' || tr.status === 'missed') && tr.sinceExit > 20;
      if (outside || finished || tr.misses > this.opts.maxMisses) {
        if (tr.status === 'measuring') this.finishPartial(tr);
        this.retired.push(tr);
        this.tracks.splice(i, 1);
      }
    }
  }

  /** Vehicles that left the view (kept for the results list). */
  readonly retired: GunTrack[] = [];

  private finishPartial(tr: GunTrack) {
    tr.finalKmh = this.partialReading(tr);
    tr.status = tr.finalKmh === null ? 'missed' : 'measured';
  }

  private updateSummary() {
    const all = [...this.tracks, ...this.retired];
    let best: GunTrack | null = null;
    for (const tr of all) {
      // Only certified readings (full zone crossings) count towards the maximum.
      const reading = tr.status === 'measured' && tr.fullCrossing ? tr.finalKmh : null;
      if (reading !== null && (this.maxKmh === null || reading > this.maxKmh)) this.maxKmh = reading;
    }
    if (this.cal.mode === 'object') {
      const tr = this.tracks[0];
      if (tr?.currentKmh != null) {
        this.currentKmh = tr.currentKmh;
        this.currentId = tr.id;
        // Only count readings backed by a full window, so a start-up spike is not "max".
        const w = lastWindow(tr.samples, this.opts.windowSec);
        if (w.length && w[w.length - 1].t - w[0].t >= this.opts.windowSec * 0.9) {
          if (this.maxKmh === null || tr.currentKmh > this.maxKmh) this.maxKmh = tr.currentKmh;
        }
      }
      return;
    }
    // Ground mode: a vehicle in the zone right now wins; otherwise the latest finished one.
    for (const tr of this.tracks) {
      if (tr.status === 'measuring' && tr.currentKmh !== null) best = tr;
    }
    if (best) {
      this.currentKmh = best.currentKmh;
      this.currentId = best.id;
    }
  }

  /** Call when a vehicle leaves the zone to make it the "current" reading. */
  latestFinished(): GunTrack | null {
    const all = [...this.tracks, ...this.retired].filter((t) => t.status === 'measured');
    return all.length ? all[all.length - 1] : null;
  }
}
