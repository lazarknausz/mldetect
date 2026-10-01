/**
 * Speed-gun mathematics: distance over time.
 *
 * A LIDAR speed gun fires laser pulses at a vehicle and times how long each echo takes
 * to come back. That gives the distance to the vehicle at each pulse; the speed is how
 * much that distance changes per unit of time:
 *
 *     speed = Δdistance / Δtime
 *
 * We cannot fire lasers from a browser, but a video gives us the same two ingredients:
 *
 *   Δtime     — every video frame is exactly 1 / FPS seconds after the previous one, so
 *               between frame n₁ and frame n₂ the time is (n₂ − n₁) / FPS.
 *   Δdistance — how many pixels the vehicle moved, converted to metres with the
 *               calibration line the user draws over something of known length:
 *               metres per pixel = (real length of the line) / (line length in pixels).
 *
 * Like a real gun, which averages many pulses, we never trust a single frame: one pixel
 * of measurement noise between two frames 1/30 s apart is already ~10 km/h. Speeds are
 * therefore the slope of a straight line fitted through *many* (time, distance) samples
 * (least squares), which averages the noise away.
 */

export interface Point {
  x: number;
  y: number;
}

export const MS_TO_KMH = 3.6;

/** Length of a line in pixels (Pythagoras). */
export function lineLengthPx(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/**
 * The calibration ratio: how many real-world metres one pixel represents.
 * Example: a 12.19 m lane-marking cycle that is 152 px long on screen → 0.0802 m/px.
 */
export function metresPerPixel(realLengthMetres: number, lengthPx: number): number {
  if (!(realLengthMetres > 0) || !(lengthPx > 0)) throw new Error('Calibration needs a positive length');
  return realLengthMetres / lengthPx;
}

/** Time between two frames: t = (frames apart) / FPS, seconds. One frame = 1 / FPS. */
export function frameInterval(fps: number, frames = 1): number {
  return frames / fps;
}

/** v = d / t, in metres per second. */
export function speedMs(distanceMetres: number, seconds: number): number {
  return distanceMetres / seconds;
}

/** m/s → km/h (1 m/s = 3600 m / 1000 h = 3.6 km/h). */
export function toKmh(metresPerSecond: number): number {
  return metresPerSecond * MS_TO_KMH;
}

/**
 * Speed from a single pair of frames — the literal "distance over time" formula:
 *   (pixel distance × metres per pixel) / (1 / FPS) × 3.6  →  km/h.
 * Used for illustration and tests; the gun reports the noise-averaged `fitSpeed`.
 */
export function instantSpeedKmh(pixelDistance: number, mPerPx: number, fps: number, frames = 1): number {
  return toKmh(speedMs(pixelDistance * mPerPx, frameInterval(fps, frames)));
}

export interface Sample {
  /** Time, seconds (frame index / FPS). */
  t: number;
  /** Distance travelled since the vehicle was first seen, metres (one axis). */
  d: number;
}

/**
 * Least-squares slope of distance against time — the average speed over the samples,
 * robust to the per-frame jitter. Returns metres per second (signed) and its standard
 * error, or null with fewer than 3 samples.
 */
export function fitSlope(samples: readonly Sample[]): { slope: number; stderr: number } | null {
  const n = samples.length;
  if (n < 3) return null;
  let mt = 0;
  let md = 0;
  for (const s of samples) {
    mt += s.t;
    md += s.d;
  }
  mt /= n;
  md /= n;
  let stt = 0;
  let std = 0;
  for (const s of samples) {
    stt += (s.t - mt) ** 2;
    std += (s.t - mt) * (s.d - md);
  }
  if (stt < 1e-12) return null;
  const slope = std / stt;
  let sse = 0;
  for (const s of samples) sse += (s.d - md - slope * (s.t - mt)) ** 2;
  return { slope, stderr: Math.sqrt(sse / Math.max(1, n - 2) / stt) };
}

/** Samples of the last `windowSec` seconds. */
export function lastWindow<T extends { t: number }>(samples: readonly T[], windowSec: number): T[] {
  if (!samples.length) return [];
  const tEnd = samples[samples.length - 1].t;
  let i = samples.length - 1;
  while (i > 0 && tEnd - samples[i - 1].t <= windowSec + 1e-9) i--;
  return samples.slice(i);
}

/**
 * Speed (m/s) of a 2-D track from its last `windowSec` seconds: fit x(t) and y(t)
 * separately (metres), the speed is the length of the fitted velocity vector.
 */
export function fitSpeed2D(
  samples: ReadonlyArray<{ t: number; x: number; y: number }>,
  windowSec: number,
): number | null {
  const w = lastWindow(samples, windowSec);
  const fx = fitSlope(w.map((s) => ({ t: s.t, d: s.x })));
  const fy = fitSlope(w.map((s) => ({ t: s.t, d: s.y })));
  if (!fx || !fy) return null;
  return Math.hypot(fx.slope, fy.slope);
}

const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 100, 119.88, 120];

/**
 * Frame rate from the presentation times of consecutive frames (as reported by
 * `requestVideoFrameCallback`). The browser does not expose a video's FPS directly.
 * Timestamps are often rounded (WebM stores whole milliseconds: 33, 34, 33 … ms) and
 * the browser may skip frames while playing, so: estimate one frame's duration from the
 * median gap, count how many frames each gap spans, and divide the frame count by the
 * total time. The result snaps to a standard rate within 0.5 % (e.g. 29.97).
 */
export function detectFps(mediaTimes: readonly number[]): number | null {
  const times = [...mediaTimes].sort((a, b) => a - b);
  const gaps: number[] = [];
  for (let i = 1; i < times.length; i++) {
    const g = times[i] - times[i - 1];
    if (g > 1e-4) gaps.push(g);
  }
  if (gaps.length < 3) return null;
  const sorted = [...gaps].sort((a, b) => a - b);
  const one = sorted[Math.floor(sorted.length / 2)];
  const frames = gaps.reduce((n, g) => n + Math.max(1, Math.round(g / one)), 0);
  const fps = frames / gaps.reduce((s, g) => s + g, 0);
  const snap = COMMON_FPS.find((f) => Math.abs(f - fps) / f < 0.005);
  return snap ?? Math.round(fps * 100) / 100;
}

/** Frame number shown at media time `t` (frame n covers [n/FPS, (n+1)/FPS)). */
export function frameIndexAt(t: number, fps: number): number {
  return Math.floor(t * fps + 1e-3);
}

/** Media time to seek to so that frame `n` is shown: the middle of the frame. */
export function seekTimeForFrame(n: number, fps: number): number {
  return (n + 0.5) / fps;
}
