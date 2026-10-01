import { describe, expect, it } from 'vitest';
import { decodeYolox, iou, nms } from './postprocess';
import { rgbaToBgrChw, letterboxRatio } from './preprocess';
import type { Detection } from './types';

const det = (x1: number, y1: number, x2: number, y2: number, score = 0.9, classId = 2): Detection => ({
  x1, y1, x2, y2, score, classId,
});

describe('iou', () => {
  it('is 1 for identical boxes and 0 for disjoint boxes', () => {
    expect(iou(det(0, 0, 10, 10), det(0, 0, 10, 10))).toBe(1);
    expect(iou(det(0, 0, 10, 10), det(20, 20, 30, 30))).toBe(0);
  });
  it('computes partial overlap', () => {
    expect(iou(det(0, 0, 10, 10), det(5, 0, 15, 10))).toBeCloseTo(50 / 150);
  });
});

describe('nms', () => {
  it('suppresses overlapping boxes of the same class', () => {
    const kept = nms(
      [det(0, 0, 10, 10, 0.8), det(1, 1, 11, 11, 0.9), det(4, 4, 14, 14, 0.7, 7), det(50, 50, 60, 60, 0.5)],
      0.45,
    );
    expect(kept.map((d) => [d.score, d.classId])).toEqual([
      [0.9, 2],
      [0.7, 7],
      [0.5, 2],
    ]);
  });
});

describe('nms across classes', () => {
  it('drops a near-identical box with a different class but keeps distinct overlaps', () => {
    const kept = nms([det(0, 0, 100, 50, 0.9, 7), det(2, 1, 101, 52, 0.6, 5), det(0, 0, 100, 50, 0.5, 0)], 0.45);
    // truck kept; bus (IoU ≈ 0.93) dropped; same box as "person" is also dropped (IoU 1).
    expect(kept.map((d) => d.classId)).toEqual([7]);
    const kept2 = nms([det(0, 0, 100, 100, 0.9, 0), det(20, 40, 80, 100, 0.8, 1)], 0.45);
    expect(kept2).toHaveLength(2); // rider and bicycle overlap only partially
  });
});

describe('decodeYolox', () => {
  const C = 80;
  const rows = 52 * 52 + 26 * 26 + 13 * 13;
  it('decodes grid offsets, strides and the letterbox ratio', () => {
    const out = new Float32Array(rows * (5 + C));
    // Stride-8 grid cell (gx=10, gy=5).
    const r = 5 * 52 + 10;
    const o = r * (5 + C);
    out.set([0.5, 0.5, Math.log(4), Math.log(2), 0.9], o);
    out[o + 5 + 2] = 0.8; // car
    // Stride-32 cell (gx=1, gy=2): low score, should be dropped.
    const r2 = 52 * 52 + 26 * 26 + 2 * 13 + 1;
    out[r2 * (5 + C) + 4] = 0.05;

    const dets = decodeYolox(out, {
      inputSize: 416, numClasses: C, ratio: 0.5, imageWidth: 832, imageHeight: 832,
      scoreThreshold: 0.1, iouThreshold: 0.45, classFilter: null,
    });
    expect(dets).toHaveLength(1);
    const d = dets[0];
    expect(d.classId).toBe(2);
    expect(d.score).toBeCloseTo(0.72);
    // centre (10.5*8, 5.5*8) = (84, 44) model px → (168, 88) source px; size 32×16 → 64×32.
    expect((d.x1 + d.x2) / 2).toBeCloseTo(168);
    expect((d.y1 + d.y2) / 2).toBeCloseTo(88);
    expect(d.x2 - d.x1).toBeCloseTo(64);
    expect(d.y2 - d.y1).toBeCloseTo(32);
  });

  it('honours the class filter', () => {
    const out = new Float32Array(rows * (5 + C));
    out.set([0, 0, 0, 0, 0.9], 0);
    out[5 + 0] = 0.9; // person
    const base = {
      inputSize: 416, numClasses: C, ratio: 1, imageWidth: 416, imageHeight: 416,
      scoreThreshold: 0.1, iouThreshold: 0.45,
    };
    expect(decodeYolox(out, { ...base, classFilter: new Set([2]) })).toHaveLength(0);
    expect(decodeYolox(out, { ...base, classFilter: null })).toHaveLength(1);
  });
});

describe('preprocess', () => {
  it('produces BGR planar output', () => {
    const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255, 0, 0, 0, 255, 1, 2, 3, 255]);
    expect(Array.from(rgbaToBgrChw(rgba, 2))).toEqual([30, 60, 0, 3, 20, 50, 0, 2, 10, 40, 0, 1]);
  });
  it('computes the letterbox ratio from the longer side', () => {
    expect(letterboxRatio(1280, 720, 416)).toBeCloseTo(0.325);
  });
});
