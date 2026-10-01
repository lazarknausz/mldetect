import { describe, expect, it } from 'vitest';
import { buildPyramid, type GrayImage } from './globalMotion';
import { moveRect, selectFeaturesInRect, trackRegion } from './regionTracker';
import { applySim, fromSimilarity } from './transforms';

/** Smooth random texture defined everywhere (so it can be shifted by sub-pixel amounts). */
function texture(seed: number, n: number) {
  let s = seed;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const blobs = Array.from({ length: n }, () => ({
    x: rnd() * 400 - 40,
    y: rnd() * 260 - 30,
    r: 2 + rnd() * 5,
    v: (rnd() - 0.5) * 180,
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

const W = 320;
const H = 200;
const bg = texture(7, 1500);
const car = texture(99, 900);

/**
 * Frame with the background shifted by `cam` and a 90 × 50 px textured "car" whose
 * texture point (u, v) appears at centre + zoom·(u, v).
 */
function frame(cam: { x: number; y: number }, cx: number, cy: number, zoom: number): GrayImage {
  const data = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = (x - cx) / zoom;
      const v = (y - cy) / zoom;
      data[y * W + x] = Math.abs(u) < 45 && Math.abs(v) < 25 ? car(u + 150, v + 100) : bg(x - cam.x, y - cam.y);
    }
  }
  return { data, width: W, height: H };
}

describe('selectFeaturesInRect', () => {
  it('finds corners spread over the box only', () => {
    const img = frame({ x: 0, y: 0 }, 160, 100, 1);
    const pts = selectFeaturesInRect(img, { x1: 120, y1: 80, x2: 200, y2: 120 });
    expect(pts.length).toBeGreaterThan(20);
    expect(pts.every((p) => p.x >= 120 && p.x <= 200 && p.y >= 80 && p.y <= 120)).toBe(true);
  });
});

describe('trackRegion', () => {
  it('measures the object motion to a fraction of a pixel, ignoring the background', () => {
    const prev = buildPyramid(frame({ x: 0, y: 0 }, 150, 100, 1), 3);
    // Object moves (+6.37, −2.21) px and grows 3 %; the camera pans by (−1.5, 0.4) px.
    const cur = buildPyramid(frame({ x: -1.5, y: 0.4 }, 156.37, 97.79, 1.03), 3);
    const cam = fromSimilarity({ a: 1, b: 0, tx: -1.5, ty: 0.4 });
    const m = trackRegion(prev, cur, { x1: 105, y1: 75, x2: 195, y2: 125 }, { x: 5, y: -2 }, cam)!;
    expect(m).not.toBeNull();
    const [x, y] = applySim(m.transform, 150, 100);
    expect(x).toBeCloseTo(156.37, 1);
    expect(y).toBeCloseTo(97.79, 1);
    expect(Math.hypot(m.transform.a, m.transform.b)).toBeCloseTo(1.03, 2);
    const moved = moveRect({ x1: 105, y1: 75, x2: 195, y2: 125 }, m.transform);
    expect((moved.x1 + moved.x2) / 2).toBeCloseTo(156.37, 1);
    expect(moved.x2 - moved.x1).toBeCloseTo(90 * 1.03, 0);
  });

  it('returns null for a featureless region', () => {
    const flat: GrayImage = { data: new Float32Array(W * H).fill(100), width: W, height: H };
    const p = buildPyramid(flat, 3);
    expect(trackRegion(p, p, { x1: 100, y1: 50, x2: 200, y2: 150 })).toBeNull();
  });
});
