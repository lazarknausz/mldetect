import { describe, expect, it } from 'vitest';
import { framesToCsv } from './csv';
import type { TrackSnapshot } from '../tracking/types';

describe('framesToCsv', () => {
  it('writes a header and one row per track per frame', () => {
    const s = {
      id: 3, classId: 2, label: 'car', score: 0.9, state: 'confirmed', t: 1, firstSeen: 0,
      cx: 10, cy: 20, w: 30, h: 15, vx: 100, vy: 0, screenVx: 100, screenVy: 0, vw: 0, vh: 0, speedPx: 100, speedKmh: 54.04,
      headingDeg: 90, compass: 'E', stationary: false, measuring: false, approach: null, turnRate: 0, trail: [],
    } satisfies TrackSnapshot;
    const csv = framesToCsv([{ t: 1, tracks: [s] }, { t: 1.1, tracks: [] }]).trim().split('\n');
    expect(csv).toHaveLength(2);
    expect(csv[0]).toMatch(/^time_s,track_id,class/);
    expect(csv[1]).toBe('1.000,3,"car",confirmed,0.900,10.0,20.0,30.0,15.0,100.0,0.0,100.0,54.0,90,E,');
  });
});
