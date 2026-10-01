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

/** Assumed diagonal field of view of the camera (typical phone / action camera). */
export const DEFAULT_DIAGONAL_FOV_DEG = 75;

/** Focal length in pixels for a frame, from the assumed diagonal field of view. */
export function focalLengthPx(frameWidth: number, frameHeight: number, diagFovDeg = DEFAULT_DIAGONAL_FOV_DEG): number {
  return Math.hypot(frameWidth, frameHeight) / 2 / Math.tan(((diagFovDeg / 2) * Math.PI) / 180);
}

/**
 * Rate of change of log box size, d(ln s)/dt with s = √(w·h), by least squares over the
 * window. Boxes clipped by the frame edge are skipped (their size is not the object's).
 */
export function fitLogSizeRate(
  history: readonly HistoryPoint[],
  windowSec: number,
  targetPoints = 6,
): { rate: number; stderr: number } | null {
  const pts: Array<{ t: number; y: number }> = [];
  const tEnd = history.length ? history[history.length - 1].t : 0;
  for (let i = history.length - 1; i >= 0; i--) {
    const p = history[i];
    const age = tEnd - p.t;
    if (age > 2.5 * windowSec || (age > windowSec && pts.length >= targetPoints)) break;
    if (!p.clipped && p.w > 0 && p.h > 0) pts.push({ t: p.t, y: 0.5 * Math.log(p.w * p.h) });
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
  const rate = sty / stt;
  let sse = 0;
  for (const p of pts) sse += (p.y - my - rate * (p.t - mt)) ** 2;
  const stderr = Math.sqrt(sse / (n - 2) / stt);
  return { rate, stderr };
}

/** Unit 3-D direction of travel in camera coordinates (x right, y down, z away from camera). */
export type Direction3 = [number, number, number];

/**
 * Pixels per metre for one observed box. The object is a length × cross block whose
 * long axis points along its 3-D direction of travel `d`. Its projected box is
 * w = k(L|dx| + X√(1−dx²)), h = k(L|dy| + X√(1−dy²)), solved for k by least squares.
 * (Side-on: w = kL, h = kX. Head-on: w = h = kX. Top-down, moving up/down: w = kX, h = kL.)
 */
export function pixelsPerMetre(w: number, h: number, d: Direction3, ref: ReferenceSize): number {
  if (ref.kind === 'height') return h / ref.height;
  const dx = Math.min(1, Math.abs(d[0]));
  const dy = Math.min(1, Math.abs(d[1]));
  const a = ref.length * dx + ref.cross * Math.sqrt(1 - dx * dx);
  const b = ref.length * dy + ref.cross * Math.sqrt(1 - dy * dy);
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
}

export interface SpeedInput {
  history: readonly HistoryPoint[];
  /** Image velocity of the box centre, px/s. */
  vx: number;
  vy: number;
  /** Current box centre, px. */
  cx: number;
  cy: number;
  frameWidth: number;
  frameHeight: number;
  ref: ReferenceSize;
  windowSec: number;
  focalPx?: number;
  maxSamples?: number;
}

/**
 * Estimates real-world speed with a pinhole camera model. Image motion gives the
 * velocity parallel to the image plane; the growth/shrink rate of the box ("looming")
 * gives the velocity towards/away from the camera, Ż/Z = −ṡ/s. The resulting 3-D
 * direction decides which box dimension reflects the object's length, which fixes
 * the scale (m/px) and hence the speed.
 */
export function estimateSpeed(input: SpeedInput): SpeedEstimate | null {
  const { history, vx, vy, frameWidth, frameHeight, ref, windowSec } = input;
  if (!history.length) return null;
  const f = input.focalPx ?? focalLengthPx(frameWidth, frameHeight);
  const u = input.cx - frameWidth / 2;
  const v = input.cy - frameHeight / 2;

  // Only trust looming when it is clearly above the box-size noise.
  const loom = fitLogSizeRate(history, windowSec);
  const gz = loom && Math.abs(loom.rate) > 2.5 * loom.stderr ? -loom.rate : 0; // Ż / Z
  // Velocity divided by depth Z (scale-free).
  const nx = (vx + u * gz) / f;
  const ny = (vy + v * gz) / f;
  const nz = gz;
  const nn = Math.hypot(nx, ny, nz);
  const last = history[history.length - 1];
  const dir: Direction3 =
    nn > 1e-9 ? [nx / nn, ny / nn, nz / nn] : last.w >= last.h ? [1, 0, 0] : [0, 1, 0];

  const ks: number[] = [];
  const maxSamples = input.maxSamples ?? 8;
  for (let i = history.length - 1; i >= 0 && ks.length < maxSamples; i--) {
    if (!history[i].clipped) ks.push(pixelsPerMetre(history[i].w, history[i].h, dir, ref));
  }
  // Only seen cut off by the frame edge so far: the box is a lower bound on size.
  if (!ks.length) ks.push(pixelsPerMetre(last.w, last.h, dir, ref));
  const k = median(ks);
  if (!(k > 0)) return null;
  const mpp = 1 / k;
  const Z = f * mpp;
  return { metresPerSecond: Z * nn, metresPerPixel: mpp, direction: dir, depthComponent: dir[2] };
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
