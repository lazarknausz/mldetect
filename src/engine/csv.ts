import type { FrameResult } from './TrackingEngine';

const HEADER = [
  'time_s', 'track_id', 'class', 'state', 'confidence', 'x_px', 'y_px', 'width_px', 'height_px',
  'vx_px_s', 'vy_px_s', 'speed_px_s', 'speed_kmh_est', 'heading_deg', 'compass', 'approach',
];

const num = (v: number | null, digits: number) => (v === null || !Number.isFinite(v) ? '' : v.toFixed(digits));

/** One row per track per analysed frame. */
export function framesToCsv(frames: readonly FrameResult[]): string {
  const rows = [HEADER.join(',')];
  for (const f of frames) {
    for (const s of f.tracks) {
      rows.push(
        [
          num(f.t, 3), s.id, `"${s.label}"`, s.state, num(s.score, 3),
          num(s.cx, 1), num(s.cy, 1), num(s.w, 1), num(s.h, 1),
          num(s.vx, 1), num(s.vy, 1), num(s.speedPx, 1), num(s.speedKmh, 1),
          num(s.headingDeg, 0), s.compass ?? '', s.approach ?? '',
        ].join(','),
      );
    }
  }
  return rows.join('\n') + '\n';
}
