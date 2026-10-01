import { describe, expect, it } from 'vitest';
import {
  detectFps,
  fitSlope,
  fitSpeed2D,
  frameIndexAt,
  frameInterval,
  instantSpeedKmh,
  lineLengthPx,
  metresPerPixel,
  seekTimeForFrame,
  toKmh,
} from './speedMath';

describe('distance over time', () => {
  it('follows the spec formula', () => {
    // A 12 m line drawn 150 px long → 0.08 m/px.
    const mpp = metresPerPixel(12, lineLengthPx({ x: 0, y: 0 }, { x: 90, y: 120 }));
    expect(mpp).toBeCloseTo(0.08);
    // 12.5 px per frame at 30 fps: 1 m per 1/30 s = 30 m/s = 108 km/h.
    expect(instantSpeedKmh(12.5, mpp, 30)).toBeCloseTo(108);
    expect(frameInterval(29.97)).toBeCloseTo(1 / 29.97);
    expect(toKmh(10)).toBe(36);
  });

  it('averages per-frame jitter away with a least-squares fit', () => {
    let seed = 3;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
    // 30 m/s, sampled at 30 fps for 0.5 s, each position ±0.15 m (≈ 2 px) off.
    const samples = Array.from({ length: 16 }, (_, i) => ({ t: i / 30, d: 30 * (i / 30) + 0.15 * rnd() }));
    const perFrame = samples.slice(1).map((s, i) => (s.d - samples[i].d) * 30);
    const spread = Math.max(...perFrame) - Math.min(...perFrame);
    expect(spread).toBeGreaterThan(10); // single frames: ±5 m/s or more
    expect(fitSlope(samples)!.slope).toBeCloseTo(30, 0);
    const xy = samples.map((s) => ({ t: s.t, x: s.d * 0.6, y: s.d * 0.8 }));
    expect(fitSpeed2D(xy, 1)!).toBeCloseTo(30, 0);
  });
});

describe('frame timing', () => {
  it('detects common frame rates from presentation times', () => {
    const ntsc = Array.from({ length: 20 }, (_, i) => (i * 1001) / 30000);
    expect(detectFps(ntsc)).toBe(29.97);
    // A skipped frame does not change the answer.
    const skipped = ntsc.filter((_, i) => i !== 7 && i !== 12);
    expect(detectFps(skipped)).toBe(29.97);
    expect(detectFps(Array.from({ length: 10 }, (_, i) => i / 25))).toBe(25);
    expect(detectFps([0, 0.1])).toBeNull();
    // WebM rounds timestamps to whole milliseconds (33, 34, 33 … ms).
    const ms = Array.from({ length: 25 }, (_, i) => Math.round((i * 1001) / 30) / 1000);
    expect(detectFps(ms)).toBe(29.97);
  });
  it('maps frames to seek times and back', () => {
    for (const n of [0, 1, 299, 4500]) expect(frameIndexAt(seekTimeForFrame(n, 29.97), 29.97)).toBe(n);
  });
});
