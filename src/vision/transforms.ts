/**
 * Image-to-image transforms for camera motion: similarities (shift + rotation + zoom)
 * and plane-to-plane projective transforms (homographies: 3×3, row-major, h[8] = 1).
 * A camera that turns (pan / tilt) moves the image by a homography, not just a shift:
 * points near the edges move more than points in the middle ("keystone"). Using the
 * shift measured over the whole frame for an object in the middle would be ~10 %
 * off for a fast pan.
 */
export interface Point {
  x: number;
  y: number;
}

/** x' = a·x − b·y + tx, y' = b·x + a·y + ty (rotation + uniform scale + shift). */
export interface Similarity {
  a: number;
  b: number;
  tx: number;
  ty: number;
}

export function applySim(s: Similarity, x: number, y: number): [number, number] {
  return [s.a * x - s.b * y + s.tx, s.b * x + s.a * y + s.ty];
}

/** `outer ∘ inner`: first `inner`, then `outer`. */
export function composeSim(outer: Similarity, inner: Similarity): Similarity {
  const [tx, ty] = applySim(outer, inner.tx, inner.ty);
  return {
    a: outer.a * inner.a - outer.b * inner.b,
    b: outer.a * inner.b + outer.b * inner.a,
    tx,
    ty,
  };
}

export function invertSim(s: Similarity): Similarity {
  const d = s.a * s.a + s.b * s.b;
  const a = s.a / d;
  const b = -s.b / d;
  return { a, b, tx: -(a * s.tx - b * s.ty), ty: -(b * s.tx + a * s.ty) };
}

export function simScale(s: Similarity): number {
  return Math.hypot(s.a, s.b);
}

export type Homography = readonly [number, number, number, number, number, number, number, number, number];

export const H_IDENTITY: Homography = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function normalise(m: number[]): Homography {
  const k = Math.abs(m[8]) > 1e-12 ? m[8] : 1;
  return m.map((v) => v / k) as unknown as Homography;
}

export function fromSimilarity(s: Similarity): Homography {
  return [s.a, -s.b, s.tx, s.b, s.a, s.ty, 0, 0, 1];
}

export function applyH(H: Homography, x: number, y: number): [number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/** `outer ∘ inner`: first `inner`, then `outer`. */
export function composeH(outer: Homography, inner: Homography): Homography {
  const m: number[] = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      m.push(outer[3 * r] * inner[c] + outer[3 * r + 1] * inner[3 + c] + outer[3 * r + 2] * inner[6 + c]);
    }
  }
  return normalise(m);
}

export function invertH(H: Homography): Homography {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return normalise([
    A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det,
  ]);
}

/** Jacobian [∂x'/∂x, ∂x'/∂y, ∂y'/∂x, ∂y'/∂y] at (x, y). */
export function jacobianH(H: Homography, x: number, y: number): [number, number, number, number] {
  const w = H[6] * x + H[7] * y + H[8];
  const X = (H[0] * x + H[1] * y + H[2]) / w;
  const Y = (H[3] * x + H[4] * y + H[5]) / w;
  return [(H[0] - X * H[6]) / w, (H[1] - X * H[7]) / w, (H[3] - Y * H[6]) / w, (H[4] - Y * H[7]) / w];
}

/** Maps a velocity / small displacement at (x, y). */
export function applyHLinear(H: Homography, x: number, y: number, vx: number, vy: number): [number, number] {
  const J = jacobianH(H, x, y);
  return [J[0] * vx + J[1] * vy, J[2] * vx + J[3] * vy];
}

/** Local zoom factor at (x, y) (√ of the area change). */
export function localScaleH(H: Homography, x: number, y: number): number {
  const J = jacobianH(H, x, y);
  return Math.sqrt(Math.abs(J[0] * J[3] - J[1] * J[2]));
}

/** Same transform in coordinates multiplied by `k` (e.g. a downscaled image → source px). */
export function rescaleH(H: Homography, k: number): Homography {
  return normalise([H[0], H[1], H[2] * k, H[3], H[4], H[5] * k, H[6] / k, H[7] / k, H[8]]);
}

/** A fraction `r` of a near-identity motion (for extrapolating it over a gap). */
export function scaleMotionH(H: Homography, r: number): Homography {
  return normalise(H.map((v, i) => (i % 4 === 0 ? 1 + (v - 1) * r : v * r)));
}

/** Largest displacement of the frame corners. */
export function cornerMotion(H: Homography, width: number, height: number): number {
  let m = 0;
  for (const [x, y] of [[0, 0], [width, 0], [0, height], [width, height]]) {
    const [u, v] = applyH(H, x, y);
    m = Math.max(m, Math.hypot(u - x, v - y));
  }
  return m;
}

function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const k = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/**
 * Least-squares fit of a similarity plus keystone,
 *   H = [[a, −b, tx], [b, a, ty], [g1, g2, 1]],
 * which is exact for a camera that pans, tilts, rolls and zooms (to first order in the
 * keystone) and stays well-conditioned for narrow (zoomed) views, unlike a free
 * 8-parameter homography. Falls back to `fallback` if the keystone is implausible.
 */
export function fitCameraHomography(
  src: readonly Point[],
  dst: readonly Point[],
  centre: Point,
  fallback: Similarity,
): Homography {
  const n = src.length;
  if (n < 4) return fromSimilarity(fallback);
  // Normalise for conditioning: centre on the image, unit RMS radius.
  let r2 = 0;
  for (const p of src) r2 += (p.x - centre.x) ** 2 + (p.y - centre.y) ** 2;
  const s = Math.sqrt(2 / Math.max(1e-9, r2 / n));
  const AtA = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
  const Atb = new Array<number>(6).fill(0);
  const add = (row: number[], rhs: number) => {
    for (let i = 0; i < 6; i++) {
      Atb[i] += row[i] * rhs;
      for (let j = 0; j < 6; j++) AtA[i][j] += row[i] * row[j];
    }
  };
  for (let i = 0; i < n; i++) {
    const x = (src[i].x - centre.x) * s;
    const y = (src[i].y - centre.y) * s;
    const X = (dst[i].x - centre.x) * s;
    const Y = (dst[i].y - centre.y) * s;
    // X(1 + g1 x + g2 y) = a x − b y + tx ;  Y(1 + g1 x + g2 y) = b x + a y + ty
    add([x, -y, 1, 0, -X * x, -X * y], X);
    add([y, x, 0, 1, -Y * x, -Y * y], Y);
  }
  // A light ridge on the keystone terms keeps them at 0 when the data cannot tell.
  AtA[4][4] += 1e-3 * n;
  AtA[5][5] += 1e-3 * n;
  const p = solve(AtA, Atb);
  if (!p || Math.hypot(p[4], p[5]) > 0.2) return fromSimilarity(fallback);
  const [a, b, tx, ty, g1, g2] = p;
  const Hn: Homography = [a, -b, tx, b, a, ty, g1, g2, 1];
  // H = T⁻¹ · Hn · T with T = [[s, 0, −s·cx], [0, s, −s·cy], [0, 0, 1]].
  const T: Homography = [s, 0, -s * centre.x, 0, s, -s * centre.y, 0, 0, 1];
  const Ti: Homography = [1 / s, 0, centre.x, 0, 1 / s, centre.y, 0, 0, 1];
  const H = composeH(Ti, composeH(Hn, T));
  // Sanity: must agree with the robust similarity on the inliers to within a few px.
  let worst = 0;
  for (let i = 0; i < n; i++) {
    const [u, v] = applyH(H, src[i].x, src[i].y);
    const [su, sv] = applySim(fallback, src[i].x, src[i].y);
    worst = Math.max(worst, Math.hypot(u - su, v - sv));
  }
  const span = Math.sqrt(r2 / n);
  return worst > 0.1 * span ? fromSimilarity(fallback) : H;
}

/**
 * Camera rotation (3×3, row-major) implied by a homography between two views with
 * focal length `f` and optical centre (cx, cy): H ≈ K·R·K⁻¹. Zoom and noise are
 * removed by orthonormalising.
 */
export function rotationFromH(H: Homography, f: number, cx: number, cy: number): number[] {
  // M = K⁻¹ · H · K
  const K = [f, 0, cx, 0, f, cy, 0, 0, 1];
  const Ki = [1 / f, 0, -cx / f, 0, 1 / f, -cy / f, 0, 0, 1];
  const mul = (A: readonly number[], B: readonly number[]) => {
    const m: number[] = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m.push(A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]);
    return m;
  };
  const M = mul(mul(Ki, H), K);
  // Gram–Schmidt on the columns.
  const col = (j: number) => [M[j], M[3 + j], M[6 + j]];
  const nrm = (v: number[]) => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return v.map((x) => x / l);
  };
  const c0 = nrm(col(0));
  let c1 = col(1);
  const d01 = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
  c1 = nrm(c1.map((x, i) => x - d01 * c0[i]));
  const c2 = [c0[1] * c1[2] - c0[2] * c1[1], c0[2] * c1[0] - c0[0] * c1[2], c0[0] * c1[1] - c0[1] * c1[0]];
  return [c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]];
}

/** Rᵀ·B for 3×3 row-major matrices. */
export function transposeMul(R: readonly number[], B: readonly number[]): number[] {
  const m: number[] = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m.push(R[r] * B[c] + R[3 + r] * B[3 + c] + R[6 + r] * B[6 + c]);
  return m;
}
