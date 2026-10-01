import type { ReferenceSize } from '../detection/classes';
import type { HistoryPoint } from './types';

export interface Velocity {
  vx: number;
  vy: number;
}

/**
 * Least-squares velocity of the box centre over the last `windowSec` seconds of
 * history. Returns null if there is not enough data for a stable fit.
 */
export function fitVelocity(
  history: readonly HistoryPoint[],
  windowSec: number,
  minPoints = 3,
  minSpan = 0.15,
  /** At low frame rates, stretch the window (up to 2.5×) to include this many points. */
  targetPoints = 5,
): Velocity | null {
  if (history.length < minPoints) return null;
  const tEnd = history[history.length - 1].t;
  let start = history.length - 1;
  while (start > 0) {
    const age = tEnd - history[start - 1].t;
    const n = history.length - start;
    if (age <= windowSec || (n < targetPoints && age <= 2.5 * windowSec)) start--;
    else break;
  }
  return fitRange(history, start, history.length, minPoints, minSpan);
}

function fitRange(
  h: readonly HistoryPoint[],
  from: number,
  to: number,
  minPoints: number,
  minSpan: number,
): Velocity | null {
  const n = to - from;
  if (n < minPoints || h[to - 1].t - h[from].t < minSpan) return null;
  let st = 0;
  let sx = 0;
  let sy = 0;
  for (let i = from; i < to; i++) {
    st += h[i].t;
    sx += h[i].cx;
    sy += h[i].cy;
  }
  const mt = st / n;
  const mx = sx / n;
  const my = sy / n;
  let stt = 0;
  let stx = 0;
  let sty = 0;
  for (let i = from; i < to; i++) {
    const dt = h[i].t - mt;
    stt += dt * dt;
    stx += dt * (h[i].cx - mx);
    sty += dt * (h[i].cy - my);
  }
  if (stt <= 1e-9) return null;
  return { vx: stx / stt, vy: sty / stt };
}

/**
 * Estimates how fast the direction of travel is rotating (rad/s), by comparing the
 * velocity in the older and newer halves of the window. Returns 0 when unreliable.
 */
export function estimateTurnRate(history: readonly HistoryPoint[], windowSec: number): number {
  if (history.length < 6) return 0;
  const tEnd = history[history.length - 1].t;
  let start = history.length - 1;
  while (start > 0 && tEnd - history[start - 1].t <= windowSec) start--;
  const n = history.length - start;
  // Need a well-filled window: short or sparse histories make spurious curves.
  if (n < 8 || tEnd - history[start].t < 0.75 * windowSec) return 0;
  const mid = start + Math.floor(n / 2);
  const a = fitRange(history, start, mid, 3, 0.05);
  const b = fitRange(history, mid, history.length, 3, 0.05);
  if (!a || !b) return 0;
  const sizeRef = Math.max(history[history.length - 1].w, history[history.length - 1].h, 1);
  // Only trust the turn when the object moves clearly (≥ 0.5 body lengths/s).
  if (Math.hypot(a.vx, a.vy) < 0.5 * sizeRef || Math.hypot(b.vx, b.vy) < 0.5 * sizeRef) return 0;
  const tA = (history[start].t + history[mid - 1].t) / 2;
  const tB = (history[mid].t + history[history.length - 1].t) / 2;
  if (tB - tA <= 1e-6) return 0;
  let dAng = Math.atan2(b.vy, b.vx) - Math.atan2(a.vy, a.vx);
  while (dAng > Math.PI) dAng -= 2 * Math.PI;
  while (dAng < -Math.PI) dAng += 2 * Math.PI;
  const omega = dAng / (tB - tA);
  if (Math.abs(omega) < 0.1) return 0;
  return Math.max(-1.5, Math.min(1.5, omega));
}

/** Screen bearing in degrees (0 = up/north, 90 = right/east). */
export function headingDegrees(vx: number, vy: number): number {
  const deg = (Math.atan2(vx, -vy) * 180) / Math.PI;
  return (deg + 360) % 360;
}

const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

export function compassLabel(headingDeg: number): string {
  return COMPASS[Math.round(headingDeg / 45) % 8];
}

export function median(values: readonly number[]): number {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Diagonal field of view of a typical phone / action camera main lens ("1×"). */
export const DEFAULT_DIAGONAL_FOV_DEG = 75;

/** Diagonal field of view of the main lens zoomed in `zoom` times. */
export function zoomToDiagonalFov(zoom: number): number {
  const half = (DEFAULT_DIAGONAL_FOV_DEG / 2) * (Math.PI / 180);
  return (2 * Math.atan(Math.tan(half) / zoom) * 180) / Math.PI;
}

/** Focal length in pixels for a frame, from the assumed diagonal field of view. */
export function focalLengthPx(frameWidth: number, frameHeight: number, diagFovDeg = DEFAULT_DIAGONAL_FOV_DEG): number {
  return Math.hypot(frameWidth, frameHeight) / 2 / Math.tan(((diagFovDeg / 2) * Math.PI) / 180);
}

/**
 * Current rate of change of log box size, d(ln s)/dt with s = √(w·h), at the latest
 * sample. Distance is proportional to 1/s, and for an object moving at constant
 * velocity 1/s changes linearly with time, so a line is fitted to 1/s (by least
 * squares over the window) rather than to ln s, whose slope drifts as the distance
 * changes. Boxes clipped by the frame edge are skipped (their size is not the object's).
 */
export function fitLogSizeRate(
  history: readonly HistoryPoint[],
  windowSec: number,
  targetPoints = 6,
  /** Apparent size of a sample (any quantity proportional to 1 / distance). */
  sizeOf: (p: HistoryPoint) => number = (p) => Math.sqrt(p.w * p.h),
): { rate: number; stderr: number } | null {
  const pts: Array<{ t: number; y: number }> = [];
  const tEnd = history.length ? history[history.length - 1].t : 0;
  let s0 = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    const p = history[i];
    const age = tEnd - p.t;
    if (age > 2.5 * windowSec || (age > windowSec && pts.length >= targetPoints)) break;
    if (p.clipped || !(p.w > 0 && p.h > 0)) continue;
    const size = sizeOf(p);
    if (!(size > 0)) continue;
    if (!s0) s0 = size;
    pts.push({ t: p.t - tEnd, y: s0 / size });
  }
  const n = pts.length;
  if (n < 4) return null;
  const mt = pts.reduce((a, p) => a + p.t, 0) / n;
  const my = pts.reduce((a, p) => a + p.y, 0) / n;
  let stt = 0;
  let sty = 0;
  for (const p of pts) {
    stt += (p.t - mt) ** 2;
    sty += (p.t - mt) * (p.y - my);
  }
  if (stt < 1e-6) return null;
  const slope = sty / stt;
  const y0 = my - slope * mt; // value of the line now (t = 0)
  if (!(y0 > 0.05)) return null;
  let sse = 0;
  for (const p of pts) sse += (p.y - my - slope * (p.t - mt)) ** 2;
  const stderr = Math.sqrt(sse / (n - 2) / stt) / y0;
  // d(ln s)/dt = −d(ln(1/s))/dt
  return { rate: -slope / y0, stderr };
}

/** Unit 3-D direction of travel in camera coordinates (x right, y down, z away from camera). */
export type Direction3 = [number, number, number];

const ROLL_STEPS = 36;
/**
 * Expected relative squared box-size error of a detector for a w × h box (as in the
 * residual of `estimateSpeed`): ≈ 3 % of the size (measured on YOLOX-Tiny), but
 * never below ~1 px.
 */
function boxNoiseResidual(w: number, h: number): number {
  return 2 * (0.03 ** 2 + 1 / (w * w + h * h));
}
/** F-statistic needed to accept a depth (towards / away) component. */
const DEPTH_F_THRESHOLD = 6;
/**
 * Implied change of distance over the samples below which depth motion is not even
 * considered (box-size noise), and from which on it is accepted (between the two the
 * speed is held / "measuring").
 */
const MIN_DEPTH_EVIDENCE = 0.03;
const MIN_DEPTH_CHANGE = 0.15;
/** With this much history, a smaller (but significant) change is accepted too, seconds. */
const DEPTH_DECISION_SEC = 1.5;

/** One observed box, with its centre's normalised offset from the optical centre (u/f, v/f). */
export interface BoxView {
  w: number;
  h: number;
  ox: number;
  oy: number;
  /** Rotation (3×3 row-major) from the camera that saw this box to the current one. */
  rot?: readonly number[];
}

/** A current-camera direction expressed in the camera of `v` (Rᵀ·d). */
function directionIn(v: BoxView, d: Direction3): Direction3 {
  const R = v.rot;
  if (!R) return d;
  return [
    R[0] * d[0] + R[3] * d[1] + R[6] * d[2],
    R[1] * d[0] + R[4] * d[1] + R[7] * d[2],
    R[2] * d[0] + R[5] * d[1] + R[8] * d[2],
  ];
}

type MotionRef = Extract<ReferenceSize, { kind: 'motion' }>;

/**
 * The block's three axes projected onto the image x / y axes at one image position
 * (before any roll about the travel direction): travel `d`, reference up `u`, side `s`.
 */
interface ProjectedAxes {
  xd: number;
  yd: number;
  xu: number;
  yu: number;
  xs: number;
  ys: number;
}

function projectAxes(d: Direction3, ox: number, oy: number): ProjectedAxes {
  // Reference "up": image-up made perpendicular to d (or towards the camera when d is
  // itself vertical on screen); "side" completes the frame.
  let ux = d[1] * d[0];
  let uy = -1 + d[1] * d[1];
  let uz = d[1] * d[2];
  let un = Math.hypot(ux, uy, uz);
  if (un < 0.2) {
    ux = d[2] * d[0];
    uy = d[2] * d[1];
    uz = -1 + d[2] * d[2];
    un = Math.hypot(ux, uy, uz);
  }
  ux /= un;
  uy /= un;
  uz /= un;
  const sx = d[1] * uz - d[2] * uy;
  const sy = d[2] * ux - d[0] * uz;
  const sz = d[0] * uy - d[1] * ux;
  return {
    xd: d[0] - ox * d[2],
    yd: d[1] - oy * d[2],
    xu: ux - ox * uz,
    yu: uy - oy * uz,
    xs: sx - ox * sz,
    ys: sy - oy * sz,
  };
}

/**
 * Projected extent (a, b) of the block per unit of scale along the image x / y axes,
 * rolled by angle φ (given as cos / sin) about its travel direction. Rolled up =
 * c·up + sn·side, rolled side = −sn·up + c·side.
 */
function extentOf(p: ProjectedAxes, ref: MotionRef, c: number, sn: number): [number, number] {
  return [
    ref.length * Math.abs(p.xd) + ref.height * Math.abs(c * p.xu + sn * p.xs) + ref.width * Math.abs(c * p.xs - sn * p.xu),
    ref.length * Math.abs(p.yd) + ref.height * Math.abs(c * p.yu + sn * p.ys) + ref.width * Math.abs(c * p.ys - sn * p.yu),
  ];
}

function blockExtent(d: Direction3, ref: MotionRef, ox: number, oy: number, phi: number): [number, number] {
  return extentOf(projectAxes(d, ox, oy), ref, Math.cos(phi), Math.sin(phi));
}

/**
 * Roll of the block about its travel direction that best explains the shapes of the
 * given boxes (one roll for all: it depends on the viewpoint, which changes slowly),
 * with a weak preference for "upright on screen" when the shapes cannot tell.
 */
export function fitRoll(
  views: readonly BoxView[],
  d: Direction3,
  ref: ReferenceSize,
  steps = ROLL_STEPS,
  refineIters = 10,
): number {
  if (ref.kind !== 'motion' || !views.length) return 0;
  const axes = views.map((v) => projectAxes(directionIn(v, d), v.ox, v.oy));
  const norms = views.map((v) => 1 / (v.w * v.w + v.h * v.h));
  const cost = (phi: number) => {
    const c = Math.cos(phi);
    const sn = Math.sin(phi);
    let misfit = 0;
    for (let i = 0; i < views.length; i++) {
      const [a, b] = extentOf(axes[i], ref, c, sn);
      const v = views[i];
      misfit += ((v.w * b - v.h * a) ** 2 / (a * a + b * b)) * norms[i];
    }
    return misfit / views.length + 0.005 * phi * phi;
  };
  const step = Math.PI / steps;
  let bestPhi = 0;
  let best = Infinity;
  for (let i = 0; i < steps; i++) {
    const phi = -Math.PI / 2 + i * step;
    const c = cost(phi);
    if (c < best) {
      best = c;
      bestPhi = phi;
    }
  }
  // Golden-section refinement around the best grid point.
  let lo = bestPhi - step;
  let hi = bestPhi + step;
  const g = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - g * (hi - lo);
  let x2 = lo + g * (hi - lo);
  let f1 = cost(x1);
  let f2 = cost(x2);
  for (let it = 0; it < refineIters; it++) {
    if (f1 < f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - g * (hi - lo);
      f1 = cost(x1);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + g * (hi - lo);
      f2 = cost(x2);
    }
  }
  const refined = (lo + hi) / 2;
  return cost(refined) < best ? refined : bestPhi;
}

/**
 * Pixels per metre for one observed box.
 *
 * The object is a length × width × height block travelling along its length axis,
 * in the 3-D direction `d` (camera coordinates: x right, y down, z forward). A small
 * 3-D displacement δ at depth Z, seen at normalised image offset (ox, oy) = (u/f, v/f)
 * from the optical centre, moves the image point by (f/Z)·(jx·δ, jy·δ) with
 * jx = (1, 0, −ox), jy = (0, 1, −oy). So with k = f/Z (pixels per metre) the box is
 *   w = k(L|jx·d| + W|jx·s| + H|jx·u|),  h = k(L|jy·d| + W|jy·s| + H|jy·u|)
 * where s (side) and u (up) are the block's other two axes. Their roll about d depends
 * on how the camera is tilted; it is fitted to the box shape (`fitRoll`) unless given.
 * Then k is solved by least squares.
 *
 * So a plane crossing the view is scaled by its length whether it is seen from the
 * side or from below, a car seen from behind by its width and height, and a car below
 * an overpass camera by the part of its length that the slanted view reveals.
 */
export function pixelsPerMetre(
  w: number,
  h: number,
  d: Direction3,
  ref: ReferenceSize,
  ox = 0,
  oy = 0,
  roll?: number,
): number {
  if (ref.kind === 'height') return h / ref.height;
  if (!(w * w + h * h > 0)) return 0;
  const phi = roll ?? fitRoll([{ w, h, ox, oy }], d, ref);
  const [a, b] = blockExtent(d, ref, ox, oy, phi);
  return (w * a + h * b) / (a * a + b * b);
}

export interface SpeedEstimate {
  /** Real-world speed, m/s. */
  metresPerSecond: number;
  /** Metres per pixel at the object's distance. */
  metresPerPixel: number;
  direction: Direction3;
  /** Component of travel towards (−) or away from (+) the camera, −1…1. */
  depthComponent: number;
  /** False while it is not yet clear whether the object moves towards / away from the camera. */
  settled: boolean;
}

export interface SpeedInput {
  /** Box history in the same (camera-motion compensated) coordinates as the centre. */
  history: readonly HistoryPoint[];
  /** Principal point (optical centre), px. Defaults to the frame centre. */
  principalX?: number;
  principalY?: number;
  frameWidth: number;
  frameHeight: number;
  ref: ReferenceSize;
  /** Window for the position fit, seconds. */
  windowSec: number;
  /** Window for the growth/shrink rate (needs longer than position to beat box noise). */
  loomWindowSec?: number;
  focalPx?: number;
  maxSamples?: number;
}

/**
 * Estimates real-world speed with a pinhole camera model, at the time of the latest
 * observation.
 *
 * - Depth: the box size s is inversely proportional to the distance Z, so the
 *   growth/shrink rate ("looming") gives Ż/Z = −ṡ/s, and Z(t)/Z(now) for every past
 *   sample.
 * - Sideways: X = u·Z/f, so u·Z(t)/Z(now) moves linearly in time for an object at
 *   constant velocity even while it approaches or recedes (its on-screen speed does
 *   not). Its slope is f·Ẋ/Z(now).
 * - Scale: the 3-D direction says which box dimension shows the object's length,
 *   which gives metres per pixel (and hence Z) now.
 */
export function estimateSpeed(input: SpeedInput): SpeedEstimate | null {
  const { history, frameWidth, frameHeight, ref, windowSec } = input;
  if (!history.length) return null;
  const f = input.focalPx ?? focalLengthPx(frameWidth, frameHeight);
  const px = input.principalX ?? frameWidth / 2;
  const py = input.principalY ?? frameHeight / 2;
  const last = history[history.length - 1];
  const tNow = last.t;
  const loomWindow = input.loomWindowSec ?? Math.max(windowSec, 2);

  // Each box is judged from the camera that saw it (`view`); `toCurrent` converts its
  // pixels-per-metre into the current camera's (same range, different depth axis).
  type Sample = { p: HistoryPoint; v: BoxView; toCurrent: number };
  const sampleOf = (p: HistoryPoint): Sample => {
    const v: BoxView = { w: p.w, h: p.h, ox: ((p.ix ?? p.cx) - px) / f, oy: ((p.iy ?? p.cy) - py) / f, rot: p.rot };
    const oc = Math.hypot((p.cx - px) / f, (p.cy - py) / f);
    return { p, v, toCurrent: Math.sqrt(1 + oc * oc) / Math.sqrt(1 + v.ox * v.ox + v.oy * v.oy) };
  };
  const samples = history.map(sampleOf);
  const recent = samples.filter((s) => !s.p.clipped && tNow - s.p.t <= loomWindow).slice(-30);
  if (!recent.length) recent.push(sampleOf(last));
  const recentViews = recent.map((s) => s.v);
  const kOf = (s: Sample, dir: Direction3, roll: number) =>
    pixelsPerMetre(s.v.w, s.v.h, directionIn(s.v, dir), ref, s.v.ox, s.v.oy, roll) * s.toCurrent;
  /** Z(t)/Z(now) for a depth rate (kept positive). */
  const depthRatioOf = (rate: number) => (t: number) => Math.max(0.2, 1 + rate * (tNow - t));

  // A cheaper, subsampled fit is used while searching over depth rates.
  const every = Math.max(1, Math.ceil(recent.length / 12));
  const coarse = recent.filter((_, i) => (recent.length - 1 - i) % every === 0);

  /**
   * Full motion hypothesis for a depth rate (Ż/Z = −rate): direction, roll, speed
   * (per unit depth) and how well it explains the box sizes and shapes.
   */
  // Sideways fit over the position window (as `fitVelocity`), on precomputed arrays.
  let first = history.length - 1;
  while (first > 0) {
    const age = tNow - history[first - 1].t;
    const n = history.length - first;
    if (age <= windowSec || (n < 5 && age <= 2.5 * windowSec)) first--;
    else break;
  }
  const win = history.slice(first);
  if (win.length < 3 || tNow - win[0].t < 0.15) return null;
  const lateralFit = (rate: number): Velocity | null => {
    const n = win.length;
    let st = 0;
    let sx = 0;
    let sy = 0;
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = win[i];
      const r = Math.max(0.2, 1 + rate * (tNow - p.t));
      xs[i] = (p.cx - px) * r;
      ys[i] = (p.cy - py) * r;
      st += p.t;
      sx += xs[i];
      sy += ys[i];
    }
    const mt = st / n;
    let stt = 0;
    let stx = 0;
    let sty = 0;
    for (let i = 0; i < n; i++) {
      const dt = win[i].t - mt;
      stt += dt * dt;
      stx += dt * (xs[i] - sx / n);
      sty += dt * (ys[i] - sy / n);
    }
    return stt > 1e-9 ? { vx: stx / stt, vy: sty / stt } : null;
  };

  const hypothesis = (rate: number, fine: boolean) => {
    const ratio = depthRatioOf(rate);
    const lateral = lateralFit(rate);
    if (!lateral) return null;
    const nx = lateral.vx / f; // Ẋ / Z(now)
    const ny = lateral.vy / f;
    const nz = -rate; // Ż / Z(now)
    const nn = Math.hypot(nx, ny, nz);
    const dir: Direction3 =
      nn > 1e-9 ? [nx / nn, ny / nn, nz / nn] : last.w >= last.h ? [1, 0, 0] : [0, 1, 0];
    const set = fine ? recent : coarse;
    const roll = fine ? fitRoll(recentViews, dir, ref) : fitRoll(set.map((sm) => sm.v), dir, ref, 12, 6);
    // Residual of the predicted box sizes (both dimensions) with one common scale.
    const ext = set.map((sm) => {
      const [a, b] =
        ref.kind === 'motion' ? blockExtent(directionIn(sm.v, dir), ref, sm.v.ox, sm.v.oy, roll) : [0, ref.height];
      return { sm, a, b, r: ratio(sm.p.t) * sm.toCurrent };
    });
    let num = 0;
    let den = 0;
    for (const e of ext) {
      num += (e.sm.v.w * e.a + e.sm.v.h * e.b) / e.r;
      den += (e.a * e.a + e.b * e.b) / (e.r * e.r);
    }
    const kNow = den > 0 ? num / den : 0;
    let residual = 0;
    for (const e of ext) {
      const { w, h } = e.sm.v;
      residual += ((w - (kNow * e.a) / e.r) ** 2 + (h - (kNow * e.b) / e.r) ** 2) / (w * w + h * h);
    }
    return { rate, dir, roll, speed: nn, residual: residual / ext.length };
  };

  // Depth motion ("looming"). The raw growth of the box is a first guess, but a box
  // also changes size and shape when the viewing angle changes (a car approaching an
  // overpass shows more of its roof; a plane turning towards the camera shows more
  // wing), so the depth rate is chosen as the one whose 3-D motion best explains all
  // the boxes, searched relative to the sideways motion.
  const flat = hypothesis(0, false);
  if (!flat) return null;
  const sideways = flat.speed;
  const candidates = new Set<number>([0]);
  const raw = fitLogSizeRate(history, loomWindow);
  if (raw) candidates.add(raw.rate);
  const viaModel = fitLogSizeRate(history, loomWindow, 6, (p) => kOf(sampleOf(p), flat.dir, flat.roll));
  if (viaModel) candidates.add(viaModel.rate);
  for (const q of [0.06, 0.12, 0.2, 0.3, 0.45, 0.7, 1, 1.5, 2.5]) {
    candidates.add(q * sideways);
    candidates.add(-q * sideways);
  }
  const sorted = [...candidates].filter(Number.isFinite).sort((a, b) => a - b);
  const scored = sorted.map((r) => hypothesis(r, false));
  let bi = 0;
  scored.forEach((h, i) => {
    if (h && (!scored[bi] || h.residual < scored[bi]!.residual)) bi = i;
  });
  // Golden-section refinement between the neighbouring candidates.
  let lo = sorted[Math.max(0, bi - 1)];
  let hi = sorted[Math.min(sorted.length - 1, bi + 1)];
  const g = (Math.sqrt(5) - 1) / 2;
  const res = (r: number) => hypothesis(r, false)?.residual ?? Infinity;
  let x1 = hi - g * (hi - lo);
  let x2 = lo + g * (hi - lo);
  let f1 = res(x1);
  let f2 = res(x2);
  for (let it = 0; it < 8 && hi - lo > 1e-4; it++) {
    if (f1 < f2) {
      hi = x2;
      x2 = x1;
      f2 = f1;
      x1 = hi - g * (hi - lo);
      f1 = res(x1);
    } else {
      lo = x1;
      x1 = x2;
      f1 = f2;
      x2 = lo + g * (hi - lo);
      f2 = res(x2);
    }
  }
  const bestRate = Math.min(f1, f2) < (scored[bi]?.residual ?? Infinity) ? (f1 < f2 ? x1 : x2) : sorted[bi];

  // Keep the depth motion only if it explains the boxes significantly better than
  // none (F-test for one extra parameter, against at least typical detector box noise).
  const best = hypothesis(bestRate, true);
  const flatFine = hypothesis(0, true);
  if (!best || !flatFine) return null;
  const dof = Math.max(1, 2 * recent.length - 3);
  const floor = recent.reduce((acc, sm) => acc + boxNoiseResidual(sm.v.w, sm.v.h), 0) / recent.length;
  const noise = Math.max(best.residual, floor);
  // Detector boxes also drift in size for a while (not just jitter), so the implied
  // change of distance over the observed span must be large too.
  const span = recent[recent.length - 1].p.t - recent[0].p.t;
  const fits =
    Math.abs(bestRate) * span >= MIN_DEPTH_EVIDENCE &&
    (flatFine.residual - best.residual) * dof > DEPTH_F_THRESHOLD * noise;
  const bigEnough = Math.abs(bestRate) * span >= MIN_DEPTH_CHANGE || span >= DEPTH_DECISION_SEC;
  const chosen = fits && bigEnough ? best : flatFine;
  const { rate, dir, roll } = chosen;
  const depthRatio = depthRatioOf(rate);

  // Scale from the recent boxes, each brought forward to "now" with the depth change
  // (an approaching object is not measured with its older, smaller size).
  const ks: number[] = [];
  const maxSamples = input.maxSamples ?? 10;
  for (let i = samples.length - 1; i >= 0 && ks.length < maxSamples; i--) {
    const sm = samples[i];
    if (!sm.p.clipped) ks.push(kOf(sm, dir, roll) * depthRatio(sm.p.t));
  }
  // Only seen cut off by the frame edge so far: the box is a lower bound on size.
  if (!ks.length) ks.push(kOf(samples[samples.length - 1], dir, roll));
  const k = median(ks);
  if (!(k > 0)) return null;
  const mpp = 1 / k;
  const Z = f * mpp;
  return {
    metresPerSecond: Z * chosen.speed,
    metresPerPixel: mpp,
    direction: dir,
    depthComponent: dir[2],
    // Boxes suggest motion towards / away from the camera but not conclusively yet.
    settled: !(fits && !bigEnough),
  };
}

export const MS_TO_KMH = 3.6;

export interface PathPoint {
  t: number;
  x: number;
  y: number;
}

/**
 * Predicts the future centre path assuming constant speed and constant turn rate
 * (CTRV). With `turnRate` = 0 this is a straight line.
 */
export function predictPath(
  cx: number,
  cy: number,
  vx: number,
  vy: number,
  turnRate: number,
  horizonSec: number,
  stepSec = 0.1,
): PathPoint[] {
  const pts: PathPoint[] = [{ t: 0, x: cx, y: cy }];
  const speed = Math.hypot(vx, vy);
  let theta = Math.atan2(vy, vx);
  let x = cx;
  let y = cy;
  const steps = Math.max(1, Math.ceil(horizonSec / stepSec));
  const dt = horizonSec / steps;
  for (let i = 1; i <= steps; i++) {
    // Midpoint integration keeps arcs accurate for coarse steps.
    const thetaMid = theta + (turnRate * dt) / 2;
    x += speed * Math.cos(thetaMid) * dt;
    y += speed * Math.sin(thetaMid) * dt;
    theta += turnRate * dt;
    pts.push({ t: i * dt, x, y });
  }
  return pts;
}
