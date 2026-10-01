export interface HistoryPoint {
  t: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  /** Kalman-smoothed centre (for drawing trails). */
  sx?: number;
  sy?: number;
  /** Box touched the frame edge, so its size is not the object's full size. */
  clipped?: boolean;
}

export type TrackState = 'tentative' | 'confirmed' | 'lost';

/** Immutable view of a track at one moment, consumed by the UI and renderer. */
export interface TrackSnapshot {
  id: number;
  classId: number;
  label: string;
  score: number;
  state: TrackState;
  /** Time (video seconds) this snapshot describes. */
  t: number;
  firstSeen: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  /** Smoothed on-screen velocity, px/s. */
  vx: number;
  vy: number;
  /** Rate of change of box size, px/s (lets predicted boxes grow/shrink). */
  vw: number;
  vh: number;
  speedPx: number;
  /** Estimated real-world speed; null when the class has no reference size. */
  speedKmh: number | null;
  /** Screen bearing: 0° = up, 90° = right. Null when stationary. */
  headingDeg: number | null;
  compass: string | null;
  stationary: boolean;
  /** Not enough observations yet for a reliable speed/direction. */
  measuring: boolean;
  /** Mostly moving towards or away from the camera (from box growth/shrink). */
  approach: 'approaching' | 'receding' | null;
  /** Turn rate of the heading, rad/s (positive = clockwise on screen). */
  turnRate: number;
  /** Recent centre positions, oldest first. */
  trail: Array<{ t: number; x: number; y: number }>;
}
