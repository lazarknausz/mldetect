import { describe, expect, it } from 'vitest';
import { CLASS_ID } from '../detection/classes';
import type { Detection } from '../detection/types';
import { focalLengthPx } from './motion';
import { Tracker } from './tracker';
import type { TrackSnapshot } from './types';

const W = 1280;
const H = 720;

function box(cx: number, cy: number, w: number, h: number, classId: number = CLASS_ID.car, score = 0.8): Detection {
  return { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score, classId };
}

/**
 * A 4.5 × 1.8 × 1.5 m car driving across the view 20 px/m away (camera with the default
 * 75° lens): off-centre, perspective also shows the width of its side in depth.
 */
function sideOnCar(cx: number, cy: number): Detection {
  const f = focalLengthPx(W, H);
  const k = 20;
  return box(cx, cy, k * (4.5 + 1.8 * Math.abs(cx - W / 2) / f), k * (1.5 + 1.8 * Math.abs(cy - H / 2) / f));
}

describe('Tracker', () => {
  it('keeps stable IDs for two crossing-lane cars and measures their speed', () => {
    const tr = new Tracker();
    const fps = 10;
    let snaps: TrackSnapshot[] = [];
    for (let i = 0; i < 30; i++) {
      const t = i / fps;
      snaps = tr.update([sideOnCar(100 + 300 * t, 300), sideOnCar(1100 - 150 * t, 450)], t, W, H);
    }
    expect(snaps).toHaveLength(2);
    expect(new Set(snaps.map((s) => s.id))).toEqual(new Set([1, 2]));
    expect(tr.totalConfirmed).toBe(2);
    const right = snaps.find((s) => s.id === 1)!;
    const left = snaps.find((s) => s.id === 2)!;
    // 4.5 m car at 20 px/m → 90 px long; 300 px/s = 15 m/s = 54 km/h.
    expect(right.speedKmh!).toBeCloseTo(54, -1);
    expect(right.compass).toBe('E');
    expect(left.speedKmh!).toBeCloseTo(27, -1);
    expect(left.compass).toBe('W');
  });

  it('survives a short occlusion and keeps the same ID', () => {
    const tr = new Tracker();
    let id: number | undefined;
    for (let i = 0; i < 40; i++) {
      const t = i / 10;
      const occluded = i >= 15 && i < 20;
      const snaps = tr.update(occluded ? [] : [box(100 + 100 * t, 300, 60, 30)], t, W, H);
      if (i === 10) id = snaps[0].id;
      if (occluded) expect(snaps[0]?.state).toBe('lost');
      if (i === 39) {
        expect(snaps).toHaveLength(1);
        expect(snaps[0].id).toBe(id);
        expect(snaps[0].state).toBe('confirmed');
      }
    }
    expect(tr.totalConfirmed).toBe(1);
  });

  it('follows fast objects at a low processing frame rate (no IoU overlap between frames)', () => {
    const tr = new Tracker();
    let snaps: TrackSnapshot[] = [];
    for (let i = 0; i < 12; i++) {
      // 40 px wide, moving 60 px per processed frame.
      snaps = tr.update([box(50 + 60 * i, 200, 40, 20, CLASS_ID.airplane)], i / 5, W, H);
    }
    expect(snaps).toHaveLength(1);
    expect(tr.totalConfirmed).toBe(1);
    expect(snaps[0].speedPx).toBeCloseTo(300, -1);
  });

  it('drops a track once it leaves the frame', () => {
    const tr = new Tracker();
    let snaps: TrackSnapshot[] = [];
    for (let i = 0; i < 30; i++) {
      const cx = 1100 + 40 * i;
      snaps = tr.update(cx < W ? [box(cx, 300, 60, 30)] : [], i / 10, W, H);
    }
    expect(snaps).toHaveLength(0);
  });

  it('marks a parked car as stationary', () => {
    const tr = new Tracker();
    let snaps: TrackSnapshot[] = [];
    for (let i = 0; i < 20; i++) {
      snaps = tr.update([box(400 + (i % 2), 300 - (i % 3), 90, 40)], i / 10, W, H);
    }
    expect(snaps[0].stationary).toBe(true);
    expect(snaps[0].speedKmh).toBe(0);
    expect(snaps[0].headingDeg).toBeNull();
  });

  it('uses low-confidence detections to keep existing tracks alive', () => {
    const tr = new Tracker();
    let snaps: TrackSnapshot[] = [];
    for (let i = 0; i < 20; i++) {
      const score = i < 5 ? 0.8 : 0.2; // detector becomes unsure (e.g. partial occlusion)
      snaps = tr.update([box(100 + 10 * i, 300, 60, 30, CLASS_ID.car, score)], i / 10, W, H);
    }
    expect(snaps).toHaveLength(1);
    expect(snaps[0].state).toBe('confirmed');
    // Low-confidence detections never start new tracks.
    const fresh = new Tracker();
    expect(fresh.update([box(100, 100, 60, 30, CLASS_ID.car, 0.2)], 0, W, H)).toHaveLength(0);
  });
});

describe('Tracker speed reliability', () => {
  it('reports "measuring" until enough observations exist', () => {
    const tr = new Tracker();
    const first = tr.update([box(100, 300, 90, 34, CLASS_ID.car, 0.95)], 0, W, H);
    expect(first[0].measuring).toBe(true);
    expect(first[0].speedKmh).toBeNull();
    let snaps: TrackSnapshot[] = [];
    for (let i = 1; i < 6; i++) snaps = tr.update([box(100 + 30 * i, 300, 90, 34)], i / 10, W, H);
    expect(snaps[0].measuring).toBe(false);
    expect(snaps[0].speedKmh).not.toBeNull();
  });

  it('withholds physically implausible speeds', () => {
    const tr = new Tracker();
    let snaps: TrackSnapshot[] = [];
    // A 10 px "car" moving 600 px/s ≈ 970 km/h at that scale.
    for (let i = 0; i < 8; i++) snaps = tr.update([box(100 + 60 * i, 300, 10, 4)], i / 10, W, H);
    expect(snaps[0].speedKmh).toBeNull();
    expect(snaps[0].measuring).toBe(true);
  });
});

describe('Tracker duplicate guard', () => {
  it('does not spawn a second track for a re-classified box on the same object', () => {
    const tr = new Tracker();
    let snaps: TrackSnapshot[] = [];
    for (let i = 0; i < 10; i++) {
      const dets = [box(400, 200 + 10 * i, 90, 150, CLASS_ID.bus, 0.8)];
      // From frame 5 the detector also reports a slightly larger "person" box on it.
      if (i >= 5) dets.push(box(402, 205 + 10 * i, 96, 160, CLASS_ID.person, 0.7));
      snaps = tr.update(dets, i / 10, W, H);
    }
    expect(snaps).toHaveLength(1);
    expect(tr.totalConfirmed).toBe(1);
  });
});
