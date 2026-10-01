import { describe, expect, it } from 'vitest';
import { CLASS_ID } from './classes';
import { dropCutAtTileEdge, mergeRegionDetections, planRegions, regionRatio } from './tiling';
import type { Detection } from './types';

const det = (x1: number, y1: number, x2: number, y2: number, score = 0.8, classId: number = CLASS_ID.car): Detection => ({
  x1, y1, x2, y2, score, classId,
});

describe('planRegions', () => {
  it('only uses the full frame when tiling is off or would not magnify', () => {
    expect(planRegions(1920, 1080, 416, 'off')).toEqual([{ x: 0, y: 0, w: 1920, h: 1080 }]);
    expect(planRegions(480, 270, 416, 'standard')).toHaveLength(1);
  });

  it('covers a 1080p frame with overlapping tiles that magnify small objects', () => {
    for (const mode of ['standard', 'max'] as const) {
      const regions = planRegions(1920, 1080, 416, mode);
      const [full, ...tiles] = regions;
      expect(full).toEqual({ x: 0, y: 0, w: 1920, h: 1080 });
      expect(tiles.length).toBeGreaterThanOrEqual(2);
      const gain = regionRatio(tiles[0], 416) / regionRatio(full, 416);
      expect(gain).toBeGreaterThan(mode === 'max' ? 2.5 : 1.5);
      // Every pixel is inside some tile, and neighbouring tiles overlap.
      for (const [x, y] of [[0, 0], [1919, 1079], [960, 540], [1919, 0], [0, 1079]]) {
        expect(tiles.some((t) => x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h)).toBe(true);
      }
      for (const t of tiles) {
        expect(t.x + t.w).toBeLessThanOrEqual(1920);
        expect(t.y + t.h).toBeLessThanOrEqual(1080);
      }
      const xs = [...new Set(tiles.map((t) => t.x))].sort((a, b) => a - b);
      for (let i = 1; i < xs.length; i++) expect(xs[i]).toBeLessThan(xs[i - 1] + tiles[0].w);
    }
  });
});

describe('dropCutAtTileEdge', () => {
  it('drops boxes cut by an inner tile border but keeps those at the frame edge', () => {
    const tile = { x: 840, y: 0, w: 1080, h: 1080 }; // right half of a 1920 × 1080 frame
    const kept = dropCutAtTileEdge(
      [det(841, 500, 900, 530), det(1000, 500, 1060, 530), det(1880, 500, 1920, 530)],
      tile, 1920, 1080,
    );
    expect(kept.map((d) => d.x1)).toEqual([1000, 1880]);
  });
});

describe('mergeRegionDetections', () => {
  it('keeps one box per object seen in several views', () => {
    const merged = mergeRegionDetections([det(100, 100, 140, 120, 0.5)], [det(101, 100, 141, 121, 0.7)], 0.45);
    expect(merged).toHaveLength(1);
    expect(merged[0].score).toBe(0.7);
  });

  it('drops a tile box that is just part of a larger full-frame object', () => {
    const truck = det(100, 100, 400, 250, 0.8, CLASS_ID.truck);
    const cab = det(300, 110, 395, 245, 0.6, CLASS_ID.car);
    expect(mergeRegionDetections([truck], [cab], 0.45)).toEqual([truck]);
  });

  it('adds small objects only the tiles could see', () => {
    const merged = mergeRegionDetections([det(100, 100, 400, 250)], [det(900, 300, 915, 306, 0.6)], 0.45);
    expect(merged).toHaveLength(2);
  });
});
