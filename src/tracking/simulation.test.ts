/**
 * End-to-end speed accuracy: objects are rendered as 3-D boxes through a pinhole
 * camera (the detector would see the 2-D bounding box of the projected corners), with
 * box jitter like a real detector's, and fed to the tracker frame by frame.
 */
import { describe, expect, it } from 'vitest';
import { CLASS_ID } from '../detection/classes';
import type { Detection } from '../detection/types';
import type { Homography } from '../vision/transforms';
import { focalLengthPx, zoomToDiagonalFov } from './motion';
import { Tracker } from './tracker';
import type { TrackSnapshot } from './types';

type V3 = [number, number, number];
const add = (a: V3, b: V3, k = 1): V3 => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V3): V3 => {
  const l = Math.hypot(...a);
  return [a[0] / l, a[1] / l, a[2] / l];
};
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

const W = 1920;
const H = 1080;

/** Camera at the origin. World: X right, Y up, Z forward (horizontal). */
interface Camera {
  /** Rotation about the vertical axis (positive = turn right), rad. */
  yaw: number;
  /** Rotation up (positive = look up), rad. */
  pitch: number;
  zoom: number;
}

function cameraAxes(c: Camera) {
  const fwd: V3 = [Math.sin(c.yaw) * Math.cos(c.pitch), Math.sin(c.pitch), Math.cos(c.yaw) * Math.cos(c.pitch)];
  const right: V3 = [Math.cos(c.yaw), 0, -Math.sin(c.yaw)];
  const up = cross(fwd, right);
  return { fwd, right, up, f: focalLengthPx(W, H, zoomToDiagonalFov(c.zoom)) };
}

function project(c: Camera, p: V3): [number, number] | null {
  const { fwd, right, up, f } = cameraAxes(c);
  const z = dot(p, fwd);
  if (z <= 0.1) return null;
  return [W / 2 + (f * dot(p, right)) / z, H / 2 - (f * dot(p, up)) / z];
}

/** Bounding box of a length × width × height block heading along `heading`. */
function boxOf(c: Camera, centre: V3, heading: V3, dims: V3): Detection | null {
  const fwd = norm(heading);
  const side = norm(cross([0, 1, 0], fwd));
  const up = cross(fwd, side);
  let x1 = Infinity;
  let y1 = Infinity;
  let x2 = -Infinity;
  let y2 = -Infinity;
  for (const a of [-0.5, 0.5]) {
    for (const b of [-0.5, 0.5]) {
      for (const e of [-0.5, 0.5]) {
        const p = add(add(add(centre, fwd, a * dims[0]), side, b * dims[1]), up, e * dims[2]);
        const q = project(c, p);
        if (!q) return null;
        x1 = Math.min(x1, q[0]);
        y1 = Math.min(y1, q[1]);
        x2 = Math.max(x2, q[0]);
        y2 = Math.max(y2, q[1]);
      }
    }
  }
  return { x1, y1, x2, y2, score: 0.9, classId: -1 };
}

/** Exact image motion between two views of a rotating camera: H = K·Rb·Raᵀ·K⁻¹. */
function cameraMotion(a: Camera, b: Camera): Homography {
  const A = cameraAxes(a);
  const B = cameraAxes(b);
  // Columns: image of a's right / down / forward axes in b's camera frame.
  const toB = (v: V3): V3 => [dot(v, B.right), -dot(v, B.up), dot(v, B.fwd)];
  const r = toB(A.right);
  const d = toB([-A.up[0], -A.up[1], -A.up[2]]);
  const fw = toB(A.fwd);
  const f = B.f;
  const fa = A.f;
  // p_b ~ K_b · [r d fw] · K_a⁻¹ · p_a with K = [[f, 0, W/2], [0, f, H/2], [0, 0, 1]].
  const M = [
    [r[0], d[0], fw[0]],
    [r[1], d[1], fw[1]],
    [r[2], d[2], fw[2]],
  ];
  const Kb = [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]];
  const KaInv = [[1 / fa, 0, -W / 2 / fa], [0, 1 / fa, -H / 2 / fa], [0, 0, 1]];
  const mul = (X: number[][], Y: number[][]) =>
    X.map((row) => [0, 1, 2].map((c) => row[0] * Y[0][c] + row[1] * Y[1][c] + row[2] * Y[2][c]));
  const Hm = mul(mul(Kb, M), KaInv);
  return Hm.flat().map((v) => v / Hm[2][2]) as unknown as Homography;
}

interface Scenario {
  classId: number;
  dims: V3;
  start: V3;
  velocity: V3;
  fps: number;
  seconds: number;
  camera: (t: number, objectAt: V3) => Camera;
  /** Relative box-edge jitter (fraction of box size) and centre jitter (px). */
  jitter?: number;
  trackerZoom?: number;
}

function run(s: Scenario): { snaps: TrackSnapshot[]; truthKmh: number; speeds: Array<{ t: number; v: number }> } {
  const tracker = new Tracker({ zoom: s.trackerZoom ?? 1 });
  let seed = 3;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
  let prevCam: Camera | null = null;
  let snaps: TrackSnapshot[] = [];
  const speeds: Array<{ t: number; v: number }> = [];
  const n = Math.round(s.seconds * s.fps);
  for (let i = 0; i < n; i++) {
    const t = i / s.fps;
    const pos = add(s.start, s.velocity, t);
    const cam = s.camera(t, pos);
    const det = boxOf(cam, pos, s.velocity, s.dims);
    const dets: Detection[] = [];
    if (det) {
      const j = s.jitter ?? 0.02;
      const bw = det.x2 - det.x1;
      const bh = det.y2 - det.y1;
      dets.push({
        x1: det.x1 + j * bw * rnd(),
        x2: det.x2 + j * bw * rnd(),
        y1: det.y1 + j * bh * rnd(),
        y2: det.y2 + j * bh * rnd(),
        score: 0.9,
        classId: s.classId,
      });
    }
    snaps = tracker.update(dets, t, W, H, prevCam ? cameraMotion(prevCam, cam) : undefined);
    if (snaps[0]?.speedKmh != null) speeds.push({ t, v: snaps[0].speedKmh });
    prevCam = cam;
  }
  return { snaps, truthKmh: Math.hypot(...s.velocity) * 3.6, speeds };
}

const level: Camera = { yaw: 0, pitch: 0, zoom: 1 };
const CAR: V3 = [4.6, 1.8, 1.45];
const A320: V3 = [37.6, 35.8, 11.8];

/**
 * Every reading after `fromSec` is within `tolerance` of the truth. Readings in the
 * first second(s) of a track may be within `earlyTolerance` (motion towards / away
 * from the camera needs some history to be measured).
 */
function expectWithin(r: ReturnType<typeof run>, tolerance: number, fromSec = 1, earlyTolerance = tolerance) {
  const steady = r.speeds.filter((s) => s.t >= fromSec);
  expect(steady.length).toBeGreaterThan(3);
  for (const { t, v } of r.speeds) {
    const err = Math.abs(v - r.truthKmh) / r.truthKmh;
    if (t >= fromSec) expect(err).toBeLessThan(tolerance);
    else if (t >= 1) expect(err).toBeLessThan(earlyTolerance);
  }
}

describe('speed accuracy (simulated 3-D scenes)', () => {
  it('car crossing the view', () => {
    const r = run({
      classId: CLASS_ID.car, dims: CAR, start: [-12, -1.5, 35], velocity: [13.9, 0, 0],
      fps: 15, seconds: 2, camera: () => level,
    });
    expectWithin(r, 0.05);
  });

  it('distant background car (about 20 px long)', () => {
    const r = run({
      classId: CLASS_ID.car, dims: CAR, start: [-20, -3, 250], velocity: [16.7, 0, 0],
      fps: 15, seconds: 2.5, camera: () => level, jitter: 0.03,
    });
    expectWithin(r, 0.08);
  });

  it('car approaching an overpass camera', () => {
    const r = run({
      classId: CLASS_ID.car, dims: CAR, start: [1.8, -8, 90], velocity: [0, 0, -27.8],
      fps: 15, seconds: 2, camera: () => ({ yaw: 0, pitch: -0.12, zoom: 1 }),
    });
    expectWithin(r, 0.15);
    expect(r.snaps[0].approach).toBe('approaching');
  });

  it('car driving away diagonally', () => {
    const r = run({
      classId: CLASS_ID.car, dims: CAR, start: [-3, -3, 15], velocity: [8, 0, 18],
      fps: 15, seconds: 2, camera: () => ({ yaw: 0, pitch: -0.1, zoom: 1 }),
    });
    expectWithin(r, 0.1);
  });

  it('airliner crossing overhead, seen from below', () => {
    const r = run({
      classId: CLASS_ID.airplane, dims: A320, start: [-250, 400, 350], velocity: [75, 0, 0],
      fps: 15, seconds: 3, camera: () => ({ yaw: 0, pitch: 0.85, zoom: 1 }),
    });
    expectWithin(r, 0.1);
  });

  it('airliner side-on in the distance through a 3× zoom', () => {
    const r = run({
      classId: CLASS_ID.airplane, dims: A320, start: [-120, 150, 1500], velocity: [80, -3, 0],
      fps: 15, seconds: 3, camera: () => ({ yaw: 0, pitch: 0.1, zoom: 3 }), trackerZoom: 3,
    });
    expectWithin(r, 0.05);
  });

  it('airliner climbing away from the camera', () => {
    const r = run({
      classId: CLASS_ID.airplane, dims: A320, start: [30, 60, 500], velocity: [20, 12, 80],
      fps: 15, seconds: 4, camera: () => ({ yaw: 0, pitch: 0.12, zoom: 1 }),
    });
    expectWithin(r, 0.1, 2.5, 0.35);
    expect(r.snaps[0].approach).toBe('receding');
  });

  it('camera panning to keep a plane centred: speed is the plane’s, not the camera’s', () => {
    const follow = (_t: number, p: V3): Camera => ({
      yaw: Math.atan2(p[0], p[2]),
      pitch: Math.atan2(p[1], Math.hypot(p[0], p[2])),
      zoom: 1,
    });
    const r = run({
      classId: CLASS_ID.airplane, dims: A320, start: [-200, 120, 500], velocity: [70, 2, 0],
      fps: 15, seconds: 4, camera: follow,
    });
    // On screen the plane barely moves…
    expect(Math.hypot(r.snaps[0].screenVx, r.snaps[0].screenVy)).toBeLessThan(40);
    // …but its real speed is still measured.
    expectWithin(r, 0.08, 3, 0.25);
  });

  it('a parked car stays at ~0 km/h while the camera pans across it', () => {
    let yaw = -0.2;
    const r = run({
      classId: CLASS_ID.car, dims: CAR, start: [0, -1.5, 30], velocity: [1e-6, 0, 0],
      fps: 15, seconds: 2, camera: () => ({ yaw: (yaw += 0.012), pitch: 0, zoom: 1 }),
    });
    expect(r.snaps[0].stationary).toBe(true);
    expect(r.snaps[0].speedKmh).toBe(0);
  });
});
