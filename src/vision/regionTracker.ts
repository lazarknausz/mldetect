/**
 * Sub-pixel tracking of one object between two frames.
 *
 * A detector box jitters by a few pixels from frame to frame. At 30 fps one pixel of
 * jitter on a 0.1 m/px scale is already ±11 km/h, so detector boxes alone cannot give
 * a precise frame-to-frame displacement. Instead we follow the texture *inside* the
 * object (corners on the car body / plane fuselage) with pyramidal Lucas–Kanade optical
 * flow, which is accurate to ~0.1 px, and fit one motion (shift + zoom + rotation) to
 * all of those points with RANSAC so that a few bad points cannot spoil it.
 */

import { estimateGlobalMotion, fitSimilarityRansac, trackPoints, type GrayImage, type Rect } from './globalMotion';
import { applyH, applySim, rescaleH, type Homography, type Point, type Similarity } from './transforms';

/**
 * Well-textured corners inside `rect` (Shi–Tomasi minimum eigenvalue), at most one per
 * cell of a `grid × grid` lattice so that they cover the whole object.
 */
export function selectFeaturesInRect(img: GrayImage, rect: Rect, grid = 8, radius = 3): Point[] {
  const { data, width: w, height: h } = img;
  const m = radius + 2;
  const x1 = Math.max(m, Math.floor(rect.x1));
  const y1 = Math.max(m, Math.floor(rect.y1));
  const x2 = Math.min(w - 1 - m, Math.ceil(rect.x2));
  const y2 = Math.min(h - 1 - m, Math.ceil(rect.y2));
  if (x2 - x1 < 4 || y2 - y1 < 4) return [];

  // Integral images of the structure tensor over the rect (plus a margin for the window).
  const ox = x1 - radius - 1;
  const oy = y1 - radius - 1;
  const RW = x2 - x1 + 2 * radius + 3;
  const RH = y2 - y1 + 2 * radius + 3;
  const W1 = RW + 1;
  const sxx = new Float64Array(W1 * (RH + 1));
  const sxy = new Float64Array(W1 * (RH + 1));
  const syy = new Float64Array(W1 * (RH + 1));
  for (let ry = 0; ry < RH; ry++) {
    let rxx = 0;
    let rxy = 0;
    let ryy = 0;
    const y = oy + ry;
    for (let rx = 0; rx < RW; rx++) {
      const x = ox + rx;
      let gx = 0;
      let gy = 0;
      if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        const i = y * w + x;
        gx = (data[i + 1] - data[i - 1]) / 2;
        gy = (data[i + w] - data[i - w]) / 2;
      }
      rxx += gx * gx;
      rxy += gx * gy;
      ryy += gy * gy;
      const o = (ry + 1) * W1 + rx + 1;
      sxx[o] = sxx[o - W1] + rxx;
      sxy[o] = sxy[o - W1] + rxy;
      syy[o] = syy[o - W1] + ryy;
    }
  }
  const box = (s: Float64Array, x: number, y: number) => {
    const lx = x - ox;
    const ly = y - oy;
    return (
      s[(ly + radius + 1) * W1 + lx + radius + 1] -
      s[(ly - radius) * W1 + lx + radius + 1] -
      s[(ly + radius + 1) * W1 + lx - radius] +
      s[(ly - radius) * W1 + lx - radius]
    );
  };
  const area = (2 * radius + 1) ** 2;
  const minEig = 2 * area; // weaker than this is compression noise
  const cols = Math.max(1, Math.min(grid, Math.floor((x2 - x1) / 3)));
  const rows = Math.max(1, Math.min(grid, Math.floor((y2 - y1) / 3)));
  const cw = (x2 - x1) / cols;
  const ch = (y2 - y1) / rows;
  const pts: Point[] = [];
  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      let best = minEig;
      let bx = -1;
      let by = -1;
      const xs = Math.floor(x1 + gx * cw);
      const xe = Math.floor(x1 + (gx + 1) * cw);
      const ys = Math.floor(y1 + gy * ch);
      const ye = Math.floor(y1 + (gy + 1) * ch);
      for (let y = ys; y < ye; y++) {
        for (let x = xs; x < xe; x++) {
          const a = box(sxx, x, y);
          const b = box(sxy, x, y);
          const c = box(syy, x, y);
          const lam = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
          if (lam > best) {
            best = lam;
            bx = x;
            by = y;
          }
        }
      }
      if (bx >= 0) pts.push({ x: bx, y: by });
    }
  }
  return pts;
}

/** x' = a·x + b·y + tx, y' = c·x + d·y + ty — also captures stretching along one axis. */
export interface Affine {
  a: number;
  b: number;
  c: number;
  d: number;
  tx: number;
  ty: number;
}

export function applyAffine(m: Affine, x: number, y: number): [number, number] {
  return [m.a * x + m.b * y + m.tx, m.c * x + m.d * y + m.ty];
}

/**
 * Least-squares affine fit. Unlike a similarity, an affine map follows an object that
 * turns relative to the camera: its outline stretches along one direction only
 * (e.g. a plane's fuselage appearing longer as it turns side-on).
 */
export function fitAffine(src: readonly Point[], dst: readonly Point[], idx: readonly number[]): Affine | null {
  const n = idx.length;
  if (n < 3) return null;
  let mx = 0;
  let my = 0;
  let mu = 0;
  let mv = 0;
  for (const i of idx) {
    mx += src[i].x;
    my += src[i].y;
    mu += dst[i].x;
    mv += dst[i].y;
  }
  mx /= n;
  my /= n;
  mu /= n;
  mv /= n;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sux = 0;
  let suy = 0;
  let svx = 0;
  let svy = 0;
  for (const i of idx) {
    const x = src[i].x - mx;
    const y = src[i].y - my;
    const u = dst[i].x - mu;
    const v = dst[i].y - mv;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
    sux += u * x;
    suy += u * y;
    svx += v * x;
    svy += v * y;
  }
  const det = sxx * syy - sxy * sxy;
  // Points (nearly) on one line cannot fix an affine map.
  if (det < 1e-6 * (sxx + syy) ** 2 || det <= 0) return null;
  const a = (sux * syy - suy * sxy) / det;
  const b = (suy * sxx - sux * sxy) / det;
  const c = (svx * syy - svy * sxy) / det;
  const d = (svy * sxx - svx * sxy) / det;
  return { a, b, c, d, tx: mu - a * mx - b * my, ty: mv - c * mx - d * my };
}

export interface RegionMotion {
  /** Maps the object's points in the previous frame to the current frame. */
  transform: Similarity;
  /** Affine version of the same motion (null if the points could not fix one). */
  affine: Affine | null;
  inliers: number;
  tracked: number;
}

/**
 * Motion of the object inside `rect` from `prev` to `cur` (image pyramids, level-0 px).
 *
 * @param guess  expected shift of the object (e.g. its previous displacement), px
 * @param camera camera motion between the frames, if known: points that move exactly
 *               like the camera are background showing inside the box and are dropped
 */
export function trackRegion(
  prev: readonly GrayImage[],
  cur: readonly GrayImage[],
  rect: Rect,
  guess: Point = { x: 0, y: 0 },
  camera: Homography | null = null,
  /** Only use features for which this returns true (e.g. near a line). */
  keep?: (p: Point) => boolean,
  grid = 8,
): RegionMotion | null {
  // Shrink the box a little: its border is mostly background.
  const mx = keep ? 0 : 0.08 * (rect.x2 - rect.x1);
  const my = keep ? 0 : 0.08 * (rect.y2 - rect.y1);
  const inner = { x1: rect.x1 + mx, y1: rect.y1 + my, x2: rect.x2 - mx, y2: rect.y2 - my };
  let pts = selectFeaturesInRect(prev[0], inner, grid);
  if (keep) pts = pts.filter(keep);
  if (pts.length < 5) return null;
  const fwd = trackPoints(prev, cur, pts, (p) => ({ x: p.x + guess.x, y: p.y + guess.y }));
  const back = trackPoints(
    cur,
    prev,
    fwd.map((q, i) => q ?? pts[i]),
    (q) => ({ x: q.x - guess.x, y: q.y - guess.y }),
  );
  let src: Point[] = [];
  let dst: Point[] = [];
  for (let i = 0; i < pts.length; i++) {
    const q = fwd[i];
    const r = back[i];
    // Forward–backward consistency: a point that does not come back is unreliable.
    if (!q || !r || Math.hypot(r.x - pts[i].x, r.y - pts[i].y) > 0.5) continue;
    src.push(pts[i]);
    dst.push(q);
  }
  if (src.length < 5) return null;

  if (camera) {
    // Background visible inside the box moves exactly like the camera. If enough points
    // move differently, keep only those (they are the object).
    const own = src.map((p, i) => {
      const [x, y] = applyH(camera, p.x, p.y);
      return Math.hypot(dst[i].x - x, dst[i].y - y) > 0.75;
    });
    const n = own.filter(Boolean).length;
    if (n >= 5 && n >= 0.3 * src.length) {
      src = src.filter((_, i) => own[i]);
      dst = dst.filter((_, i) => own[i]);
    }
  }

  const fit = fitSimilarityRansac(src, dst, 0.6, 200);
  if (!fit || fit.inliers.length < Math.max(4, 0.4 * src.length)) return null;
  // Affine refinement: start from points that roughly agree with the similarity (a
  // turning object deviates from it by more than 0.6 px at its ends), then keep the ones
  // that fit the affine map tightly.
  let affine: Affine | null = null;
  let loose: number[] = [];
  for (let i = 0; i < src.length; i++) {
    const [x, y] = applySim(fit.transform, src[i].x, src[i].y);
    if (Math.hypot(x - dst[i].x, y - dst[i].y) <= 2) loose.push(i);
  }
  for (let it = 0; it < 3 && loose.length >= 6; it++) {
    affine = fitAffine(src, dst, loose);
    if (!affine) break;
    const next: number[] = [];
    for (let i = 0; i < src.length; i++) {
      const [x, y] = applyAffine(affine, src[i].x, src[i].y);
      if (Math.hypot(x - dst[i].x, y - dst[i].y) <= 0.6) next.push(i);
    }
    if (next.length < 6) break;
    loose = next;
  }
  if (affine && loose.length < Math.max(6, 0.4 * src.length)) affine = null;
  return { transform: fit.transform, affine, inliers: fit.inliers.length, tracked: src.length };
}

/** Moves a box by a similarity (shift + zoom; rotation only moves its centre). */
export function moveRect(r: Rect, s: Similarity): Rect {
  const cx = (r.x1 + r.x2) / 2;
  const cy = (r.y1 + r.y2) / 2;
  const [nx, ny] = applySim(s, cx, cy);
  const k = Math.hypot(s.a, s.b);
  const hw = ((r.x2 - r.x1) / 2) * k;
  const hh = ((r.y2 - r.y1) / 2) * k;
  return { x1: nx - hw, y1: ny - hh, x2: nx + hw, y2: ny + hh };
}

export interface FlowTarget {
  id: number;
  /** Box in the previous frame, in pyramid level-0 pixels. */
  box: Rect;
  /** Expected own shift of the target since the previous frame, level-0 px. */
  guess: Point;
  /**
   * A line on the target (e.g. the calibration line along a fuselage). Its motion is fitted
   * from features in a narrow band around the line only, so wings or wheels — which turn
   * differently in depth — do not distort how the line stretches.
   */
  line?: { a: Point; b: Point };
}

/**
 * One frame of the speed gun's motion measurement: the camera's motion (from the
 * background, at half resolution) and each target's own motion (full resolution).
 * `exclude` lists everything in the previous frame that may move by itself.
 */
export function measureFrameMotion(
  prev: readonly GrayImage[],
  cur: readonly GrayImage[],
  targets: readonly FlowTarget[],
  exclude: readonly Rect[],
): {
  camera: Homography | null;
  targets: Array<{ id: number; motion: RegionMotion | null; lineMotion: RegionMotion | null }>;
} {
  const half = [...exclude, ...targets.map((t) => t.box)].map((r) => ({ x1: r.x1 / 2, y1: r.y1 / 2, x2: r.x2 / 2, y2: r.y2 / 2 }));
  // A whole-frame fit: half resolution is plenty precise and 4× cheaper.
  const m = prev.length > 1 && cur.length > 1 ? estimateGlobalMotion(prev.slice(1), cur.slice(1), half) : null;
  const camera = m ? rescaleH(m.transform, 2) : null;
  const out = targets.map((tg) => {
    // Start Lucas–Kanade where the target should be: camera motion + its own last step.
    let gx = tg.guess.x;
    let gy = tg.guess.y;
    if (camera) {
      const cx = (tg.box.x1 + tg.box.x2) / 2;
      const cy = (tg.box.y1 + tg.box.y2) / 2;
      const [px, py] = applyH(camera, cx, cy);
      gx += px - cx;
      gy += py - cy;
    }
    const motion = trackRegion(prev, cur, tg.box, { x: gx, y: gy }, camera);
    let lineMotion: RegionMotion | null = null;
    if (tg.line) {
      const { a, b } = tg.line;
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      const band = Math.max(6, 0.09 * len);
      const ux = (b.x - a.x) / len;
      const uy = (b.y - a.y) / len;
      const near = (p: Point) => {
        const along = (p.x - a.x) * ux + (p.y - a.y) * uy;
        const across = Math.abs(-(p.x - a.x) * uy + (p.y - a.y) * ux);
        return along >= -0.02 * len && along <= 1.02 * len && across <= band;
      };
      const rect = {
        x1: Math.min(a.x, b.x) - band,
        y1: Math.min(a.y, b.y) - band,
        x2: Math.max(a.x, b.x) + band,
        y2: Math.max(a.y, b.y) + band,
      };
      lineMotion = trackRegion(prev, cur, rect, { x: gx, y: gy }, camera, near, 14);
    }
    return { id: tg.id, motion, lineMotion };
  });
  return { camera, targets: out };
}
