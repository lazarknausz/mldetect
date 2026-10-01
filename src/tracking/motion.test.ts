import { describe, expect, it } from 'vitest';
import { BoxKalman } from './kalman';
import { CLASS_ID, REFERENCE_SIZE } from '../detection/classes';
import {
  compassLabel,
  estimateTurnRate,
  fitLogSizeRate,
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
  const car = REFERENCE_SIZE[CLASS_ID.car]!; // 4.5 × 1.8 × 1.5 m
  const k = 20; // px per metre
  it('recovers the scale for side-on, top-down and head-on views', () => {
    // Side-on: length × height.
    expect(pixelsPerMetre(90, 30, [1, 0, 0], car)).toBeCloseTo(k, 1);
    // Seen from above, driving up the screen: width × length.
    expect(pixelsPerMetre(36, 90, [0, -1, 0], car)).toBeCloseTo(k, 1);
    // Head-on: width × height.
    expect(pixelsPerMetre(36, 30, [0, 0, -1], car)).toBeCloseTo(k, 1);
  });
  it('accounts for the slanted view of an off-centre object', () => {
    // Driving away, seen 0.3 rad below the optical axis: part of the length shows.
    const k0 = pixelsPerMetre(36, 30, [0, 0, 1], car);
    const k1 = pixelsPerMetre(36, 30 + 0.3 * 4.5 * k, [0, 0, 1], car, 0, 0.3);
    expect(k1).toBeCloseTo(k0, 0);
  });
  it('tells an airliner seen from below from one seen side-on', () => {
    const plane = REFERENCE_SIZE[CLASS_ID.airplane]!; // 38 m long, 35 m span, 12 m tall
    expect(pixelsPerMetre(380, 120, [1, 0, 0], plane)).toBeCloseTo(10, 0);
    expect(pixelsPerMetre(380, 350, [1, 0, 0], plane)).toBeCloseTo(10, 0);
  });
  it('uses height for people', () => {
    expect(pixelsPerMetre(20, 170, [1, 0, 0], REFERENCE_SIZE[CLASS_ID.person]!)).toBeCloseTo(100);
  });
});

describe('fitLogSizeRate', () => {
  it('gives the current growth rate of an object approaching at constant speed', () => {
    // Z = 50 − 10 t; size ∝ 1/Z; d(ln s)/dt = 10 / Z.
    const hist: HistoryPoint[] = Array.from({ length: 20 }, (_, i) => {
      const t = i / 10;
      const s = 1000 / (50 - 10 * t);
      return { t, cx: 0, cy: 0, w: s, h: s };
    });
    const r = fitLogSizeRate(hist, 2)!;
    expect(r.rate).toBeCloseTo(10 / (50 - 10 * 1.9), 6);
  });
});
