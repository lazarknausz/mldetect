/**
 * One-dimensional constant-velocity Kalman filter (state: position, velocity).
 * A box is tracked with four independent instances (cx, cy, w, h); because the
 * motion and measurement models are block-diagonal this is equivalent to the usual
 * 8-state SORT/ByteTrack filter, without any matrix inversion.
 */
export class Kalman1D {
  p: number;
  v: number;
  // Covariance [[ppp, ppv], [ppv, pvv]]
  private ppp: number;
  private ppv = 0;
  private pvv: number;

  constructor(position: number, positionStd: number, velocityStd: number, velocity = 0) {
    this.p = position;
    this.v = velocity;
    this.ppp = positionStd ** 2;
    this.pvv = velocityStd ** 2;
  }

  /**
   * Advances the state by dt seconds. `accelStd` is the standard deviation of the
   * (white-noise) acceleration, in units/s².
   */
  predict(dt: number, accelStd: number): void {
    if (dt <= 0) return;
    const q = accelStd ** 2;
    this.p += this.v * dt;
    const ppp = this.ppp + 2 * dt * this.ppv + dt * dt * this.pvv + (q * dt ** 3) / 3;
    const ppv = this.ppv + dt * this.pvv + (q * dt * dt) / 2;
    const pvv = this.pvv + q * dt;
    this.ppp = ppp;
    this.ppv = ppv;
    this.pvv = pvv;
  }

  /** Fuses a position measurement with standard deviation `measStd`. */
  update(z: number, measStd: number): void {
    const s = this.ppp + measStd ** 2;
    const kp = this.ppp / s;
    const kv = this.ppv / s;
    const y = z - this.p;
    this.p += kp * y;
    this.v += kv * y;
    const ppp = (1 - kp) * this.ppp;
    const ppv = (1 - kp) * this.ppv;
    const pvv = this.pvv - kv * this.ppv;
    this.ppp = ppp;
    this.ppv = ppv;
    this.pvv = pvv;
  }

  get positionStd(): number {
    return Math.sqrt(this.ppp);
  }

  get velocityStd(): number {
    return Math.sqrt(this.pvv);
  }
}

export interface Box {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

/** Constant-velocity filter over a bounding box (centre + size). */
export class BoxKalman {
  readonly cx: Kalman1D;
  readonly cy: Kalman1D;
  readonly w: Kalman1D;
  readonly h: Kalman1D;

  constructor(box: Box) {
    const size = Math.max(box.w, box.h, 1);
    // Velocity is unknown at birth: allow anything up to ~30 body lengths per second.
    this.cx = new Kalman1D(box.cx, 0.1 * size, 15 * size);
    this.cy = new Kalman1D(box.cy, 0.1 * size, 15 * size);
    this.w = new Kalman1D(box.w, 0.1 * size, 1 * size);
    this.h = new Kalman1D(box.h, 0.1 * size, 1 * size);
  }

  get size(): number {
    return Math.max(this.w.p, this.h.p, 1);
  }

  predict(dt: number): void {
    const s = this.size;
    this.cx.predict(dt, 2 * s);
    this.cy.predict(dt, 2 * s);
    this.w.predict(dt, 0.5 * s);
    this.h.predict(dt, 0.5 * s);
    // Never let the box collapse or invert.
    this.w.p = Math.max(this.w.p, 2);
    this.h.p = Math.max(this.h.p, 2);
  }

  update(box: Box): void {
    const s = Math.max(box.w, box.h, 1);
    const posStd = 0.05 * s + 1;
    this.cx.update(box.cx, posStd);
    this.cy.update(box.cy, posStd);
    this.w.update(box.w, posStd);
    this.h.update(box.h, posStd);
  }

  get box(): Box {
    return { cx: this.cx.p, cy: this.cy.p, w: this.w.p, h: this.h.p };
  }

  /** Box extrapolated `dt` seconds ahead without modifying the filter. */
  boxAt(dt: number): Box {
    return {
      cx: this.cx.p + this.cx.v * dt,
      cy: this.cy.p + this.cy.v * dt,
      w: Math.max(2, this.w.p + this.w.v * dt),
      h: Math.max(2, this.h.p + this.h.v * dt),
    };
  }
}
