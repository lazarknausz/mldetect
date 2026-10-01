import { describe, expect, it } from 'vitest';
import {
  applyH,
  applySim,
  buildPyramid,
  composeSim,
  estimateGlobalMotion,
  fitSimilarityRansac,
  invertSim,
  type GrayImage,
  type Homography,
  type Similarity,
} from './globalMotion';

/** Smooth random texture (sum of blobs), defined everywhere so it can be warped. */
function texture(seed: number) {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const blobs = Array.from({ length: 4000 }, () => ({
    x: rnd() * 1400 - 300,
    y: rnd() * 900 - 250,
    r: 3 + rnd() * 10,
    v: (rnd() - 0.5) * 160,
  }));
  return (x: number, y: number) => {
    let acc = 128;
    for (const b of blobs) {
      const d2 = (x - b.x) ** 2 + (y - b.y) ** 2;
      if (d2 < 9 * b.r * b.r) acc += b.v * Math.exp(-d2 / (2 * b.r * b.r));
    }
    return acc;
  };
}

/**
 * Renders frame k of a camera that moves by `cam` per frame: pixel (x, y) of the new
 * frame shows the scene point cam⁻¹(x, y). An optional square "object" moves on its own.
 */
function render(
  scene: (x: number, y: number) => number,
  w: number,
  h: number,
  toScene: Similarity,
  object?: { x: number; y: number; size: number },
): GrayImage {
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (object && Math.abs(x - object.x) < object.size && Math.abs(y - object.y) < object.size) {
        data[y * w + x] = 40 + 30 * Math.sin((x - object.x) * 0.9) * Math.cos((y - object.y) * 0.7);
        continue;
      }
      const [sx, sy] = applySim(toScene, x, y);
      data[y * w + x] = scene(sx, sy);
    }
  }
  return { data, width: w, height: h };
}

const W = 320;
const H = 180;

function expectSim(got: Similarity | Homography, want: Similarity, tol = 0.25) {
  // Compare where the transforms send the image corners.
  for (const [x, y] of [[0, 0], [W, 0], [0, H], [W, H], [W / 2, H / 2]]) {
    const [gx, gy] = 'a' in got ? applySim(got, x, y) : applyH(got, x, y);
    const [wx, wy] = applySim(want, x, y);
    expect(Math.hypot(gx - wx, gy - wy)).toBeLessThan(tol * 4);
  }
}

describe('similarity algebra', () => {
  it('composes and inverts', () => {
    const s: Similarity = { a: 0.99, b: 0.05, tx: 12, ty: -7 };
    const id = composeSim(s, invertSim(s));
    expect(id.a).toBeCloseTo(1);
    expect(id.b).toBeCloseTo(0);
    expect(id.tx).toBeCloseTo(0);
    expect(id.ty).toBeCloseTo(0);
  });
  it('fits a transform despite outliers', () => {
    const t: Similarity = { a: 1.02, b: -0.03, tx: 5, ty: 9 };
    const src = Array.from({ length: 40 }, (_, i) => ({ x: (i * 37) % 300, y: (i * 53) % 170 }));
    const dst = src.map((p, i) => {
      const [x, y] = applySim(t, p.x, p.y);
      return i % 4 === 0 ? { x: x + 30, y: y - 12 } : { x, y };
    });
    const fit = fitSimilarityRansac(src, dst)!;
    expect(fit.inliers.length).toBe(30);
    expectSim(fit.transform, t, 0.01);
  });
});

describe('estimateGlobalMotion', () => {
  const scene = texture(7);
  const camMove = (cam: Similarity) => {
    // Frame 0 shows the scene as is; frame 1 is the camera moved by `cam`.
    const f0 = render(scene, W, H, { a: 1, b: 0, tx: 0, ty: 0 });
    const f1 = render(scene, W, H, invertSim(cam));
    return estimateGlobalMotion(buildPyramid(f0, 4), buildPyramid(f1, 4), []);
  };

  it('is the identity for a static camera', () => {
    const m = camMove({ a: 1, b: 0, tx: 0, ty: 0 })!;
    expectSim(m.transform, { a: 1, b: 0, tx: 0, ty: 0 }, 0.05);
  });

  it('measures a small pan with sub-pixel precision', () => {
    const cam = { a: 1, b: 0, tx: -3.4, ty: 1.7 };
    expectSim(camMove(cam)!.transform, cam, 0.1);
  });

  it('measures a fast pan beyond the reach of plain Lucas–Kanade', () => {
    const cam = { a: 1, b: 0, tx: -38.5, ty: 6.2 };
    expectSim(camMove(cam)!.transform, cam, 0.15);
  });

  it('measures zoom and roll', () => {
    const z = 1.03;
    const ang = 0.01;
    const a = z * Math.cos(ang);
    const b = z * Math.sin(ang);
    // Zoom about the image centre.
    const cam = { a, b, tx: W / 2 - (a * W) / 2 + (b * H) / 2, ty: H / 2 - (b * W) / 2 - (a * H) / 2 };
    expectSim(camMove(cam)!.transform, cam, 0.15);
  });

  it('ignores a large independently moving object', () => {
    const cam = { a: 1, b: 0, tx: 4, ty: 0 };
    const obj0 = { x: 160, y: 90, size: 40 };
    const obj1 = { x: 140, y: 95, size: 40 };
    const f0 = render(scene, W, H, { a: 1, b: 0, tx: 0, ty: 0 }, obj0);
    const f1 = render(scene, W, H, invertSim(cam), obj1);
    const m = estimateGlobalMotion(buildPyramid(f0, 4), buildPyramid(f1, 4), [
      { x1: 120, y1: 50, x2: 200, y2: 130 },
    ])!;
    expectSim(m.transform, cam, 0.1);
  });

  it('gives up on a featureless frame (clear sky)', () => {
    const flat = (k: number): GrayImage => ({ data: new Float32Array(W * H).fill(180 + k), width: W, height: H });
    expect(estimateGlobalMotion(buildPyramid(flat(0), 4), buildPyramid(flat(1), 4), [])).toBeNull();
  });
});
