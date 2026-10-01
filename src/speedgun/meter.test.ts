import { describe, expect, it } from 'vitest';
import type { Detection } from '../detection/types';
import type { Homography, Similarity } from '../vision/transforms';
import { SpeedMeter, type Box, type Calibration } from './meter';

const W = 1280;
const H = 720;
const FPS = 29.97;

function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
}

/** Similarity that maps box `a` onto box `b` (centre to centre, uniform zoom). */
function boxMotion(a: Box, b: Box, noise = 0): Similarity {
  const k = Math.sqrt(((b.x2 - b.x1) * (b.y2 - b.y1)) / ((a.x2 - a.x1) * (a.y2 - a.y1)));
  const ca = { x: (a.x1 + a.x2) / 2, y: (a.y1 + a.y2) / 2 };
  const cb = { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 };
  return { a: k, b: 0, tx: cb.x - k * ca.x + noise, ty: cb.y - k * ca.y - noise };
}

function jitter(b: Box, r: () => number, px: number, classId = 2): Detection {
  return { x1: b.x1 + px * r(), y1: b.y1 + px * r(), x2: b.x2 + px * r(), y2: b.y2 + px * r(), score: 0.9, classId };
}

// --------------------------------------------------------------- highway

/** Camera 9 m above the road, pitched 10° down, looking along it (f = 1000 px). */
const f = 1000;
const camH = 9;
const pitch = (10 * Math.PI) / 180;
function project(X: number, Yup: number, Z: number) {
  const zc = (camH - Yup) * Math.sin(pitch) + Z * Math.cos(pitch);
  const yc = (camH - Yup) * Math.cos(pitch) - Z * Math.sin(pitch);
  return { x: W / 2 + (f * X) / zc, y: H / 2 + (f * yc) / zc };
}
function carBox(X: number, Zc: number): Box {
  const pts = [];
  for (const dx of [-0.9, 0.9]) for (const up of [0, 1.5]) for (const dz of [-2.25, 2.25]) pts.push(project(X + dx, up, Zc + dz));
  return {
    x1: Math.min(...pts.map((p) => p.x)),
    y1: Math.min(...pts.map((p) => p.y)),
    x2: Math.max(...pts.map((p) => p.x)),
    y2: Math.max(...pts.map((p) => p.y)),
  };
}

function runHighway(cars: Array<{ X: number; Z0: number; v: number }>, frames: number) {
  // Calibration: one lane-marking cycle (12.19 m) on the left lane line, 25–37 m away.
  const cal: Calibration = { a: project(-1.8, 0, 25), b: project(-1.8, 0, 37.19), metres: 12.19, mode: 'ground' };
  const meter = new SpeedMeter(cal, W, H);
  const r = rng(11);
  const identity: Homography = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  let prev: Box[] | null = null;
  for (let n = 0; n < frames; n++) {
    const t = n / FPS;
    const boxes = cars.map((c) => carBox(c.X, c.Z0 + c.v * t));
    // The worker follows each track's own box with optical flow (±0.1 px noise).
    const motions = new Map<number, Similarity | null>();
    if (prev) {
      for (const tr of meter.tracks) {
        // Which simulated car is this track on?
        const c = { x: (tr.box.x1 + tr.box.x2) / 2, y: (tr.box.y1 + tr.box.y2) / 2 };
        const i = prev.findIndex((b) => c.x > b.x1 - 20 && c.x < b.x2 + 20 && c.y > b.y1 - 20 && c.y < b.y2 + 20);
        motions.set(tr.id, i >= 0 ? boxMotion(prev[i], boxes[i], 0.1 * r()) : null);
      }
    }
    const visible = boxes.filter((b) => b.y2 < H && b.y1 > 0);
    meter.update({
      t,
      camera: n === 0 ? undefined : identity,
      motions,
      detections: n % 3 === 0 ? visible.map((b) => jitter(b, r, 1.5)) : undefined,
    });
    prev = boxes;
  }
  return meter;
}

describe('SpeedMeter — ground (speed trap) mode', () => {
  it('measures an approaching car in another lane within 1.5 %', () => {
    const v = 31; // m/s = 111.6 km/h
    const meter = runHighway([{ X: 1.8, Z0: 70, v: -v }], 70);
    const done = [...meter.tracks, ...meter.retired].filter((t) => t.status === 'measured');
    expect(done).toHaveLength(1);
    expect(done[0].fullCrossing).toBe(true);
    expect(done[0].finalKmh!).toBeGreaterThan(111.6 * 0.985);
    expect(done[0].finalKmh!).toBeLessThan(111.6 * 1.015);
    expect(meter.maxKmh).toBeCloseTo(done[0].finalKmh!, 5);
  });

  it('measures a receding car on the far carriageway and an approaching one at once', () => {
    const meter = runHighway(
      [
        { X: -5.4, Z0: 14, v: 25 }, // 90 km/h, driving away
        { X: 5.4, Z0: 75, v: -36 }, // 129.6 km/h, approaching
      ],
      75,
    );
    const done = [...meter.tracks, ...meter.retired].filter((t) => t.status === 'measured');
    const speeds = done.map((t) => t.finalKmh!).sort((a, b) => a - b);
    expect(speeds).toHaveLength(2);
    expect(Math.abs(speeds[0] - 90) / 90).toBeLessThan(0.02);
    expect(Math.abs(speeds[1] - 129.6) / 129.6).toBeLessThan(0.02);
  });
});

// ----------------------------------------------------------------- plane

describe('SpeedMeter — object mode (camera pans and zooms to follow)', () => {
  it('measures an accelerating airliner within 2 %', () => {
    const Z = 600;
    const L = 39.5; // Boeing 737-800 length
    const fAt = (t: number) => 1500 + 125 * t; // zooming in
    const X = (t: number) => -60 + 50 * t + 1.25 * t * t; // 50 m/s, accelerating 2.5 m/s²
    const yaw = (t: number) => X(t) / Z - 0.03 * Math.sin(0.8 * t); // camera follows, imperfectly
    const planeBox = (t: number): Box => {
      const fl = fAt(t);
      const u = W / 2 + fl * (X(t) / Z - yaw(t));
      const v = H / 2 + 40;
      const w = (fl * L) / Z;
      const h = (fl * 12) / Z;
      return { x1: u - w / 2, y1: v - h / 2, x2: u + w / 2, y2: v + h / 2 };
    };
    const b0 = planeBox(0);
    const cal: Calibration = {
      a: { x: b0.x1, y: (b0.y1 + b0.y2) / 2 },
      b: { x: b0.x2, y: (b0.y1 + b0.y2) / 2 },
      metres: L,
      mode: 'object',
    };
    const meter = new SpeedMeter(cal, W, H);
    const r = rng(5);
    meter.initObject([jitter(b0, r, 1, 4)]);
    const readings: Array<{ t: number; kmh: number }> = [];
    for (let n = 0; n <= 240; n++) {
      const t = n / FPS;
      const motions = new Map<number, Similarity | null>();
      let camera: Homography | undefined;
      if (n > 0) {
        const tp = (n - 1) / FPS;
        const k = fAt(t) / fAt(tp);
        const shift = -fAt(t) * (yaw(t) - yaw(tp));
        camera = [k, 0, W / 2 - k * (W / 2) + shift, 0, k, H / 2 - k * (H / 2), 0, 0, 1];
        motions.set(meter.tracks[0].id, boxMotion(planeBox(tp), planeBox(t), 0.1 * r()));
      }
      meter.update({ t, camera, motions, detections: n % 5 === 0 ? [jitter(planeBox(t), r, 2, 4)] : undefined });
      if (meter.currentKmh !== null) readings.push({ t, kmh: meter.currentKmh });
    }
    // Compare with the true speed at the middle of the 0.5 s averaging window.
    for (const t of [1, 2, 3, 4, 5, 6, 7, 7.9]) {
      const got = readings.find((x) => x.t >= t)!;
      const truth = (50 + 2.5 * (got.t - 0.25)) * 3.6;
      expect(Math.abs(got.kmh - truth) / truth).toBeLessThan(0.02);
    }
  });
});
