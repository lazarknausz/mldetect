/**
 * Calibration state: the line the user drags over something of known length, and that
 * length in metres. Together they give the pixels-to-metres ratio:
 *
 *     metres per pixel = real length (m) / line length (px)
 *
 * The user can drag a new line anywhere, or grab either end of the existing line to
 * fine-tune it. Coordinates are stored in *video* pixels (not screen pixels), so the
 * calibration does not depend on how large the canvas is displayed.
 */

import type { Calibration, CalibrationMode } from './meter';
import { lineLengthPx, metresPerPixel, type Point } from './speedMath';

export class CalibrationState {
  a: Point | null = null;
  b: Point | null = null;
  metres: number | null = null;
  mode: CalibrationMode = 'ground';
  /** Frame the line was drawn on (measurement starts there). */
  frame = 0;
  enabled = false;
  private drag: 'a' | 'b' | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    /** Converts a pointer event to video-pixel coordinates. */
    private toVideo: (e: PointerEvent) => Point,
    /** Video pixels per screen pixel (for the grab radius). */
    private videoPerScreen: () => number,
    private onChange: () => void,
    private currentFrame: () => number,
  ) {
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    canvas.addEventListener('pointerup', this.onUp);
    canvas.addEventListener('pointercancel', this.onUp);
  }

  /** Line length on screen, video pixels. */
  get lengthPx(): number {
    return this.a && this.b ? lineLengthPx(this.a, this.b) : 0;
  }

  /** True when the line is long enough to be useful and a positive length was entered. */
  get complete(): boolean {
    return this.lengthPx >= 10 && this.metres !== null && this.metres > 0;
  }

  get ratio(): number | null {
    return this.complete ? metresPerPixel(this.metres!, this.lengthPx) : null;
  }

  toCalibration(): Calibration {
    if (!this.complete) throw new Error('Calibration incomplete');
    return { a: { ...this.a! }, b: { ...this.b! }, metres: this.metres!, mode: this.mode };
  }

  clear() {
    this.a = null;
    this.b = null;
    this.onChange();
  }

  private onDown = (e: PointerEvent) => {
    if (!this.enabled || e.button !== 0) return;
    const p = this.toVideo(e);
    const grab = 14 * this.videoPerScreen();
    // Grab an existing end point if close enough; otherwise start a new line here.
    if (this.a && Math.hypot(p.x - this.a.x, p.y - this.a.y) <= grab) this.drag = 'a';
    else if (this.b && Math.hypot(p.x - this.b.x, p.y - this.b.y) <= grab) this.drag = 'b';
    else {
      this.a = p;
      this.b = { ...p };
      this.drag = 'b';
    }
    this.canvas.setPointerCapture(e.pointerId);
    this.frame = this.currentFrame();
    this.onChange();
  };

  private onMove = (e: PointerEvent) => {
    if (!this.enabled) return;
    if (!this.drag) {
      // Show a "grab" cursor near the ends.
      const p = this.toVideo(e);
      const grab = 14 * this.videoPerScreen();
      const near = [this.a, this.b].some((q) => q && Math.hypot(p.x - q.x, p.y - q.y) <= grab);
      this.canvas.style.cursor = near ? 'grab' : 'crosshair';
      return;
    }
    const p = this.toVideo(e);
    if (this.drag === 'a') this.a = p;
    else this.b = p;
    this.onChange();
  };

  private onUp = (e: PointerEvent) => {
    if (!this.drag) return;
    this.drag = null;
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    // A click without dragging is not a line.
    if (this.lengthPx < 5) {
      this.a = null;
      this.b = null;
    }
    this.onChange();
  };
}
