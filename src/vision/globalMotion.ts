/**
 * Camera ("global") motion between two video frames.
 *
 * When the camera pans, tilts or zooms to follow an object, the object's movement on
 * screen says little about its real movement: a plane kept in the middle of the frame
 * barely moves on screen. The background, however, moves by exactly the camera's
 * motion. This module tracks corner features on the background (anything outside the
 * detected objects) with pyramidal Lucas–Kanade optical flow, finds the camera motion
 * among them with RANSAC (shift + rotation + zoom), then refines it with the keystone
 * of a turning camera (`fitCameraHomography`), so that the tracker can measure every
 * object relative to the scene instead of relative to the screen.
 */

import {
  applyH,
  applySim,
  fitCameraHomography,
  simScale,
  type Homography,
  type Point,
  type Similarity,
} from './transforms';

export * from './transforms';

// ------------------------------------------------------------------ images

export interface GrayImage {
  data: Float32Array;
  width: number;
  height: number;
}

export function rgbaToGray(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): GrayImage {
  const data = new Float32Array(width * height);
  for (let i = 0, p = 0; p < data.length; i += 4, p++) {
    data[p] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
  return { data, width, height };
}

/** Halves the resolution with a [1 2 1] binomial pre-filter. */
export function downsample(img: GrayImage): GrayImage {
  const { data: s, width: w, height: h } = img;
  const W = Math.max(1, w >> 1);
  const H = Math.max(1, h >> 1);
  const out = new Float32Array(W * H);
  const at = (x: number, y: number) =>
    s[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = 2 * x;
      const sy = 2 * y;
      let acc = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const wy = dy === 0 ? 2 : 1;
        acc += wy * (at(sx - 1, sy + dy) + 2 * at(sx, sy + dy) + at(sx + 1, sy + dy));
      }
      out[y * W + x] = acc / 16;
    }
  }
  return { data: out, width: W, height: H };
}

export function buildPyramid(img: GrayImage, levels: number): GrayImage[] {
  const pyr = [img];
  while (pyr.length < levels && pyr[pyr.length - 1].width >= 40 && pyr[pyr.length - 1].height >= 30) {
    pyr.push(downsample(pyr[pyr.length - 1]));
  }
  return pyr;
}

function sample(img: GrayImage, x: number, y: number): number {
  const { data, width: w, height: h } = img;
  if (x < 0) x = 0;
  else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0;
  else if (y > h - 1.001) y = h - 1.001;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * w + x0;
  const top = data[i] + fx * (data[i + 1] - data[i]);
  const bot = data[i + w] + fx * (data[i + w + 1] - data[i + w]);
  return top + fy * (bot - top);
}

// ---------------------------------------------------------------- features

/** Axis-aligned box in image coordinates, used to keep features off moving objects. */
export interface Rect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}


/**
 * Picks well-textured corners (Shi–Tomasi minimum eigenvalue), at most one per grid
 * cell so they are spread over the whole background, and none inside `exclude`.
 */
export function selectFeatures(
  img: GrayImage,
  exclude: readonly Rect[],
  gridCols = 24,
  gridRows = 14,
  radius = 3,
): Point[] {
  const { data, width: w, height: h } = img;
  // Integral images of the structure-tensor terms.
  const W1 = w + 1;
  const sxx = new Float64Array(W1 * (h + 1));
  const sxy = new Float64Array(W1 * (h + 1));
  const syy = new Float64Array(W1 * (h + 1));
  for (let y = 0; y < h; y++) {
    let rxx = 0;
    let rxy = 0;
    let ryy = 0;
    for (let x = 0; x < w; x++) {
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
      const o = (y + 1) * W1 + x + 1;
      sxx[o] = sxx[o - W1] + rxx;
      sxy[o] = sxy[o - W1] + rxy;
      syy[o] = syy[o - W1] + ryy;
    }
  }
  const box = (s: Float64Array, x: number, y: number) =>
    s[(y + radius + 1) * W1 + x + radius + 1] -
    s[(y - radius) * W1 + x + radius + 1] -
    s[(y + radius + 1) * W1 + x - radius] +
    s[(y - radius) * W1 + x - radius];
  const area = (2 * radius + 1) ** 2;
  // Corners weaker than this are video noise / compression artefacts (e.g. clear sky).
  const minEig = 4 * area;
  const margin = radius + 6;
  const cellW = (w - 2 * margin) / gridCols;
  const cellH = (h - 2 * margin) / gridRows;
  if (cellW < 2 || cellH < 2) return [];
  const pad = radius + 2;
  const blocked = (x: number, y: number) =>
    exclude.some((r) => x >= r.x1 - pad && x <= r.x2 + pad && y >= r.y1 - pad && y <= r.y2 + pad);

  const pts: Point[] = [];
  for (let gy = 0; gy < gridRows; gy++) {
    for (let gx = 0; gx < gridCols; gx++) {
      const xs = Math.floor(margin + gx * cellW);
      const ys = Math.floor(margin + gy * cellH);
      const xe = Math.floor(margin + (gx + 1) * cellW);
      const ye = Math.floor(margin + (gy + 1) * cellH);
      let best = minEig;
      let bx = -1;
      let by = -1;
      for (let y = ys; y < ye; y++) {
        for (let x = xs; x < xe; x++) {
          const a = box(sxx, x, y);
          const b = box(sxy, x, y);
          const c = box(syy, x, y);
          const lam = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
          if (lam > best && !blocked(x, y)) {
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

// ---------------------------------------------------------- optical flow

/**
 * Best whole-image shift at the coarsest pyramid level (sum of absolute differences),
 * so that fast pans are found even when they exceed the reach of Lucas–Kanade.
 */
export function coarseShift(prev: GrayImage, cur: GrayImage, exclude: readonly Rect[], range: number): Point {
  const { width: w, height: h } = prev;
  const idx: number[] = [];
  for (let y = range; y < h - range; y++) {
    for (let x = range; x < w - range; x++) {
      if (!exclude.some((r) => x >= r.x1 && x <= r.x2 && y >= r.y1 && y <= r.y2)) idx.push(y * w + x);
    }
  }
  if (idx.length < 50) return { x: 0, y: 0 };
  let best = Infinity;
  let bx = 0;
  let by = 0;
  for (let dy = -range; dy <= range; dy++) {
    for (let dx = -range; dx <= range; dx++) {
      const off = dy * w + dx;
      let sad = 0;
      for (let k = 0; k < idx.length && sad < best; k++) {
        sad += Math.abs(prev.data[idx[k]] - cur.data[idx[k] + off]);
      }
      // Ties (flat images) resolve to the smallest shift.
      if (sad < best - 1e-6 || (Math.abs(sad - best) <= 1e-6 && dx * dx + dy * dy < bx * bx + by * by)) {
        best = sad;
        bx = dx;
        by = dy;
      }
    }
  }
  return { x: bx, y: by };
}

/**
 * Pyramidal Lucas–Kanade: follows each point from `prev` to `cur`, starting from the
 * guess `init(point)` (in level-0 coordinates). Returns null for points that are lost.
 */
export function trackPoints(
  prev: readonly GrayImage[],
  cur: readonly GrayImage[],
  pts: readonly Point[],
  init: (p: Point) => Point,
  radius = 4,
  iterations = 12,
): Array<Point | null> {
  const levels = Math.min(prev.length, cur.length);
  return pts.map((p0) => {
    const guess0 = init(p0);
    const top = 2 ** (levels - 1);
    let gx = (guess0.x - p0.x) / top;
    let gy = (guess0.y - p0.y) / top;
    for (let L = levels - 1; L >= 0; L--) {
      const I = prev[L];
      const J = cur[L];
      const s = 2 ** L;
      const px = p0.x / s;
      const py = p0.y / s;
      let g11 = 0;
      let g12 = 0;
      let g22 = 0;
      const n = (2 * radius + 1) ** 2;
      const ix = new Float32Array(n);
      const iy = new Float32Array(n);
      const iv = new Float32Array(n);
      let k = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++, k++) {
          const x = px + dx;
          const y = py + dy;
          const gxv = (sample(I, x + 1, y) - sample(I, x - 1, y)) / 2;
          const gyv = (sample(I, x, y + 1) - sample(I, x, y - 1)) / 2;
          ix[k] = gxv;
          iy[k] = gyv;
          iv[k] = sample(I, x, y);
          g11 += gxv * gxv;
          g12 += gxv * gyv;
          g22 += gyv * gyv;
        }
      }
      const det = g11 * g22 - g12 * g12;
      const minEig = (g11 + g22) / 2 - Math.sqrt(((g11 - g22) / 2) ** 2 + g12 * g12);
      if (det < 1e-6 || minEig < 1e-3 * n) {
        if (L === 0) return null;
        gx *= 2;
        gy *= 2;
        continue;
      }
      let vx = 0;
      let vy = 0;
      for (let it = 0; it < iterations; it++) {
        let b1 = 0;
        let b2 = 0;
        k = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++, k++) {
            const diff = iv[k] - sample(J, px + dx + gx + vx, py + dy + gy + vy);
            b1 += diff * ix[k];
            b2 += diff * iy[k];
          }
        }
        const ex = (g22 * b1 - g12 * b2) / det;
        const ey = (g11 * b2 - g12 * b1) / det;
        vx += ex;
        vy += ey;
        if (ex * ex + ey * ey < 1e-4) break;
      }
      gx += vx;
      gy += vy;
      if (L > 0) {
        gx *= 2;
        gy *= 2;
      }
    }
    const x = p0.x + gx;
    const y = p0.y + gy;
    const { width: w, height: h } = cur[0];
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > w - 1 || y > h - 1) return null;
    return { x, y };
  });
}

// ------------------------------------------------------------ model fit

function fitLeastSquares(src: readonly Point[], dst: readonly Point[], idx: readonly number[]): Similarity | null {
  const n = idx.length;
  if (n < 2) return null;
  let mpx = 0;
  let mpy = 0;
  let mqx = 0;
  let mqy = 0;
  for (const i of idx) {
    mpx += src[i].x;
    mpy += src[i].y;
    mqx += dst[i].x;
    mqy += dst[i].y;
  }
  mpx /= n;
  mpy /= n;
  mqx /= n;
  mqy /= n;
  // Complex form: q̃ = z·p̃ with z = a + ib  →  z = Σ q̃·conj(p̃) / Σ |p̃|².
  let re = 0;
  let im = 0;
  let norm = 0;
  for (const i of idx) {
    const px = src[i].x - mpx;
    const py = src[i].y - mpy;
    const qx = dst[i].x - mqx;
    const qy = dst[i].y - mqy;
    re += qx * px + qy * py;
    im += qy * px - qx * py;
    norm += px * px + py * py;
  }
  if (norm < 1e-9) return null;
  const a = re / norm;
  const b = im / norm;
  return { a, b, tx: mqx - (a * mpx - b * mpy), ty: mqy - (b * mpx + a * mpy) };
}

function inliersOf(s: Similarity, src: readonly Point[], dst: readonly Point[], thresh: number): number[] {
  const out: number[] = [];
  const t2 = thresh * thresh;
  for (let i = 0; i < src.length; i++) {
    const [x, y] = applySim(s, src[i].x, src[i].y);
    if ((x - dst[i].x) ** 2 + (y - dst[i].y) ** 2 <= t2) out.push(i);
  }
  return out;
}

/** Robust similarity fit: 2-point RANSAC, then least squares on the inliers. */
export function fitSimilarityRansac(
  src: readonly Point[],
  dst: readonly Point[],
  thresh = 1,
  iterations = 150,
): { transform: Similarity; inliers: number[] } | null {
  const n = src.length;
  if (n < 2) return null;
  let seed = 12345;
  const rnd = (m: number) => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed % m;
  };
  let best: number[] = [];
  // The pure shift is a strong candidate for pans; try it explicitly.
  const shifts = src.map((p, i) => ({ x: dst[i].x - p.x, y: dst[i].y - p.y }));
  const med = (v: number[]) => [...v].sort((p, q) => p - q)[v.length >> 1];
  const shift: Similarity = { a: 1, b: 0, tx: med(shifts.map((s) => s.x)), ty: med(shifts.map((s) => s.y)) };
  best = inliersOf(shift, src, dst, thresh);
  for (let it = 0; it < iterations; it++) {
    const i = rnd(n);
    const j = rnd(n);
    if (i === j) continue;
    const s = fitLeastSquares(src, dst, [i, j]);
    if (!s) continue;
    const sc = simScale(s);
    if (sc < 0.7 || sc > 1.4) continue;
    const inl = inliersOf(s, src, dst, thresh);
    if (inl.length > best.length) best = inl;
  }
  if (best.length < 2) return null;
  let transform = fitLeastSquares(src, dst, best);
  if (!transform) return null;
  for (let refine = 0; refine < 2; refine++) {
    const inl = inliersOf(transform, src, dst, thresh);
    const t = inl.length >= 2 ? fitLeastSquares(src, dst, inl) : null;
    if (!t) break;
    transform = t;
    best = inl;
  }
  return { transform, inliers: best };
}

// ------------------------------------------------------------- top level

export interface GlobalMotion {
  /** Maps points of the previous frame to the current frame (pyramid level-0 pixels). */
  transform: Homography;
  inliers: number;
  features: number;
}

/**
 * Camera motion from `prev` to `cur` (image pyramids of the same size). `exclude` lists
 * the moving objects in `prev` so their motion is not mistaken for the camera's.
 * Returns null when the background has too little texture (e.g. clear sky).
 */
export function estimateGlobalMotion(
  prev: readonly GrayImage[],
  cur: readonly GrayImage[],
  exclude: readonly Rect[],
): GlobalMotion | null {
  const pts = selectFeatures(prev[0], exclude);
  if (pts.length < 8) return null;
  const levels = Math.min(prev.length, cur.length);
  const top = 2 ** (levels - 1);
  const scaled = exclude.map((r) => ({ x1: r.x1 / top, y1: r.y1 / top, x2: r.x2 / top, y2: r.y2 / top }));
  const shift = coarseShift(prev[levels - 1], cur[levels - 1], scaled, 8);
  const init = (p: Point) => ({ x: p.x + shift.x * top, y: p.y + shift.y * top });
  const fwd = trackPoints(prev, cur, pts, init);
  const src: Point[] = [];
  const dst: Point[] = [];
  const back = trackPoints(
    cur,
    prev,
    fwd.map((q, i) => q ?? pts[i]),
    (q) => ({ x: q.x - shift.x * top, y: q.y - shift.y * top }),
  );
  for (let i = 0; i < pts.length; i++) {
    const q = fwd[i];
    const r = back[i];
    // Forward–backward check rejects occluded / ambiguous points.
    if (!q || !r || Math.hypot(r.x - pts[i].x, r.y - pts[i].y) > 0.75) continue;
    src.push(pts[i]);
    dst.push(q);
  }
  if (src.length < 8) return null;
  // A turning camera bends the motion field (keystone), so the similarity RANSAC gets
  // a tolerance that grows with the motion; the keystone model then tightens it.
  const flows = src.map((p, i) => Math.hypot(dst[i].x - p.x, dst[i].y - p.y)).sort((a, b) => a - b);
  const fit = fitSimilarityRansac(src, dst, Math.max(1, 0.12 * flows[flows.length >> 1]));
  if (!fit || fit.inliers.length < Math.max(8, 0.35 * src.length)) return null;
  const centre = { x: prev[0].width / 2, y: prev[0].height / 2 };
  const pick = (idx: readonly number[]) => [idx.map((i) => src[i]), idx.map((i) => dst[i])] as const;
  let H = fitCameraHomography(...pick(fit.inliers), centre, fit.transform);
  let inliers: number[] = fit.inliers;
  for (let refine = 0; refine < 2; refine++) {
    const next: number[] = [];
    for (let i = 0; i < src.length; i++) {
      const [x, y] = applyH(H, src[i].x, src[i].y);
      if (Math.hypot(x - dst[i].x, y - dst[i].y) <= 1) next.push(i);
    }
    if (next.length < Math.max(8, 0.35 * src.length)) break;
    inliers = next;
    H = fitCameraHomography(...pick(inliers), centre, fitSimilarityRansac(...pick(inliers), 1e9)?.transform ?? fit.transform);
  }
  return { transform: H, inliers: inliers.length, features: pts.length };
}
