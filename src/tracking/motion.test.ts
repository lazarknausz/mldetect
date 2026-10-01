import { describe, expect, it } from 'vitest';
import { BoxKalman } from './kalman';
import { CLASS_ID, REFERENCE_SIZE } from '../detection/classes';
import {
  compassLabel,
  estimateSpeed,
  focalLengthPx,
  estimateTurnRate,
  fitVelocity,
  headingDegrees,
  pixelsPerMetre,
  predictPath,
} from './motion';
import type { HistoryPoint } from './types';

function line(n: number, fps: number, vx: number, vy: number, noise = 0): HistoryPoint[] {
  let seed = 1;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2 * noise;
  return Array.from({ length: n }, (_, i) => {
    const t = i / fps;
    return { t, cx: 100 + vx * t + rnd(), cy: 200 + vy * t + rnd(), w: 40, h: 20 };
  });
}

describe('fitVelocity', () => {
  it('recovers velocity from noisy samples', () => {
    const v = fitVelocity(line(30, 15, 120, -60, 1.5), 0.6)!;
    expect(v.vx).toBeCloseTo(120, -1);
    expect(v.vy).toBeCloseTo(-60, -1);
  });
  it('returns null without enough data', () => {
    expect(fitVelocity(line(2, 15, 10, 0), 0.6)).toBeNull();
  });
});

describe('heading', () => {
  it('uses screen compass bearings', () => {
    expect(headingDegrees(0, -1)).toBeCloseTo(0);
    expect(headingDegrees(1, 0)).toBeCloseTo(90);
    expect(headingDegrees(0, 1)).toBeCloseTo(180);
    expect(headingDegrees(-1, 0)).toBeCloseTo(270);
    expect(compassLabel(44)).toBe('NE');
    expect(compassLabel(350)).toBe('N');
  });
});

describe('turn rate & prediction', () => {
  it('is ~0 for straight motion and positive for clockwise arcs', () => {
    expect(estimateTurnRate(line(30, 15, 100, 0), 1.2)).toBe(0);
    const omega = 0.5;
    const arc: HistoryPoint[] = Array.from({ length: 30 }, (_, i) => {
      const t = i / 15;
      return { t, cx: 300 + 200 * Math.sin(omega * t), cy: 300 - 200 * Math.cos(omega * t), w: 40, h: 20 };
    });
    expect(estimateTurnRate(arc, 1.2)).toBeCloseTo(omega, 1);
  });
  it('predicts straight lines and arcs', () => {
    const straight = predictPath(0, 0, 100, 0, 0, 2);
    expect(straight.at(-1)!.x).toBeCloseTo(200);
    expect(straight.at(-1)!.y).toBeCloseTo(0);
    // Quarter circle: speed 100, omega π/4 rad/s for 2 s → radius 400/π.
    const arc = predictPath(0, 0, 100, 0, Math.PI / 4, 2, 0.01);
    const r = 400 / Math.PI;
    expect(arc.at(-1)!.x).toBeCloseTo(r, 0);
    expect(arc.at(-1)!.y).toBeCloseTo(r, 0);
  });
});

describe('BoxKalman', () => {
  it('learns a constant velocity', () => {
    const kf = new BoxKalman({ cx: 0, cy: 0, w: 40, h: 20 });
    for (let i = 1; i <= 20; i++) {
      kf.predict(0.1);
      kf.update({ cx: 25 * i, cy: -10 * i, w: 40, h: 20 });
    }
    expect(kf.cx.v).toBeCloseTo(250, -1);
    expect(kf.cy.v).toBeCloseTo(-100, -1);
    expect(kf.boxAt(1).cx).toBeCloseTo(500 + 250, -1);
  });
});

describe('scale estimation', () => {
  const car = REFERENCE_SIZE[CLASS_ID.car]!;
  const k = 20; // px per metre
  it('recovers the scale for side-on, top-down and diagonal motion', () => {
    expect(pixelsPerMetre(90, 34, [1, 0, 0], car)).toBeCloseTo(k);
    expect(pixelsPerMetre(34, 90, [0, 1, 0], car)).toBeCloseTo(k);
    const d = (k * (4.5 + 1.7)) / Math.SQRT2;
    expect(pixelsPerMetre(d, d, [Math.SQRT1_2, Math.SQRT1_2, 0], car)).toBeCloseTo(k);
    // Head-on: only the cross-section is visible.
    expect(pixelsPerMetre(34, 34, [0, 0, -1], car)).toBeCloseTo(k);
  });
  it('uses height for people', () => {
    expect(pixelsPerMetre(20, 170, [1, 0, 0], REFERENCE_SIZE[CLASS_ID.person]!)).toBeCloseTo(100);
  });
});

/** Renders a car driving along a 3-D straight line through a pinhole camera. */
function simulate(p0: [number, number, number], vel: [number, number, number], fps: number, n: number) {
  const W = 1280;
  const H = 720;
  const f = focalLengthPx(W, H);
  const car = REFERENCE_SIZE[CLASS_ID.car]!;
  const speed = Math.hypot(...vel);
  const d: [number, number, number] = [vel[0] / speed, vel[1] / speed, vel[2] / speed];
  const hist: HistoryPoint[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fps;
    const X = p0[0] + vel[0] * t;
    const Y = p0[1] + vel[1] * t;
    const Z = p0[2] + vel[2] * t;
    const k = f / Z;
    if (car.kind !== 'motion') throw new Error();
    const w = k * (car.length * Math.abs(d[0]) + car.cross * Math.sqrt(1 - d[0] ** 2));
    const h = k * (car.length * Math.abs(d[1]) + car.cross * Math.sqrt(1 - d[1] ** 2));
    hist.push({ t, cx: W / 2 + (f * X) / Z, cy: H / 2 + (f * Y) / Z, w, h });
  }
  const v = fitVelocity(hist, 0.6)!;
  const last = hist[hist.length - 1];
  return estimateSpeed({
    history: hist, vx: v.vx, vy: v.vy, cx: last.cx, cy: last.cy,
    frameWidth: W, frameHeight: H, ref: car, windowSec: 0.6,
  })!;
}

describe('estimateSpeed (pinhole model)', () => {
  it('measures a car crossing the view', () => {
    const est = simulate([-10, 2, 40], [30, 0, 0], 15, 12);
    expect(est.metresPerSecond).toBeCloseTo(30, 0);
    expect(Math.abs(est.depthComponent)).toBeLessThan(0.1);
  });
  it('measures a car driving towards an elevated camera (highway overpass)', () => {
    // Camera 8 m above the road looking along it: the car drops on screen and grows.
    const est = simulate([2, 8, 80], [0, 0, -33], 15, 12);
    expect(est.metresPerSecond).toBeGreaterThan(33 * 0.8);
    expect(est.metresPerSecond).toBeLessThan(33 * 1.2);
    expect(est.depthComponent).toBeLessThan(-0.5);
  });
  it('measures a receding car', () => {
    const est = simulate([-2, 6, 30], [0, 0, 25], 15, 12);
    expect(est.metresPerSecond).toBeGreaterThan(25 * 0.8);
    expect(est.metresPerSecond).toBeLessThan(25 * 1.2);
    expect(est.depthComponent).toBeGreaterThan(0.5);
  });
});
