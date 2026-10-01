export interface HistoryPoint {
  t: number;
  /** Box centre (in the tracker's reference frame, or the current view; see `Tracker`). */
  cx: number;
  cy: number;
  /** Box size as detected, in the frame it was detected in. */
  w: number;
  h: number;
  /**
   * Where the box was in the frame it was detected in, px, and the rotation (3×3,
   * row-major) from that frame's camera to the current one. Absent when the camera is
   * static (then they equal `cx`, `cy` and the identity). Lets the speed model judge a
   * box's shape from the angle it was actually seen at, even if the camera has turned.
   */
  ix?: number;
  iy?: number;
  rot?: readonly number[];
  /** Kalman-smoothed centre (for drawing trails). */
  sx?: number;
  sy?: number;
  /** Box touched the frame edge, so its size is not the object's full size. */
  clipped?: boolean;
}

export type TrackState = 'tentative' | 'confirmed' | 'lost';

/** `unknown`: the background has too little texture to measure camera motion. */
export type CameraState = 'static' | 'moving' | 'unknown';

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
  /**
   * The object's own velocity relative to the scene (camera motion removed), in
   * current-frame px/s. Used for heading and the predicted path.
   */
  vx: number;
  vy: number;
  /** Velocity on screen (own motion + camera motion), px/s; for drawing between frames. */
  screenVx: number;
  screenVy: number;
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
