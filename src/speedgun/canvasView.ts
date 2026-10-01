/**
 * Canvas setup and drawing. The canvas has the video's own resolution (one canvas pixel
 * = one video pixel) and is scaled to fit the page with CSS, so every coordinate in the
 * app — calibration line, boxes, distances — is in video pixels.
 */

import { CLASS_PRESETS, className } from '../detection/classes';
import type { Calibration, GunTrack, SpeedMeter } from './meter';
import { groundPoint, makeTrap } from './meter';
import type { Point } from './speedMath';

const COLORS = {
  line: '#ffd23f',
  zone: 'rgba(255, 210, 63, 0.10)',
  approaching: 'rgba(160, 175, 200, 0.9)',
  measuring: '#36d399',
  measured: '#f5b942',
  missed: 'rgba(160, 175, 200, 0.5)',
  tracking: '#36d399',
};

export class CanvasView {
  readonly ctx: CanvasRenderingContext2D;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  /** Match the canvas to the video resolution. */
  setSize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  /** Pointer position → video pixel coordinates. */
  toVideo = (e: { clientX: number; clientY: number }): Point => {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * this.canvas.width,
      y: ((e.clientY - r.top) / r.height) * this.canvas.height,
    };
  };

  /** Video pixels per CSS pixel (to keep lines and text a constant size on screen). */
  videoPerScreen = (): number => {
    const r = this.canvas.getBoundingClientRect();
    return r.width ? this.canvas.width / r.width : 1;
  };

  drawFrame(video: HTMLVideoElement) {
    this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
  }

  /** The calibration line with its real length, and (road mode) the measuring zone. */
  drawCalibration(a: Point | null, b: Point | null, metres: number | null, cal: Calibration | null) {
    const ctx = this.ctx;
    const k = this.videoPerScreen();
    if (cal && cal.mode === 'ground') {
      const trap = makeTrap(cal);
      ctx.fillStyle = COLORS.zone;
      if (trap.axis === 'y') ctx.fillRect(0, trap.lo, this.canvas.width, trap.hi - trap.lo);
      else ctx.fillRect(trap.lo, 0, trap.hi - trap.lo, this.canvas.height);
      ctx.strokeStyle = 'rgba(255, 210, 63, 0.45)';
      ctx.lineWidth = 1 * k;
      ctx.setLineDash([6 * k, 6 * k]);
      ctx.beginPath();
      for (const v of [trap.lo, trap.hi]) {
        if (trap.axis === 'y') {
          ctx.moveTo(0, v);
          ctx.lineTo(this.canvas.width, v);
        } else {
          ctx.moveTo(v, 0);
          ctx.lineTo(v, this.canvas.height);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (!a || !b) return;
    ctx.save();
    ctx.strokeStyle = COLORS.line;
    ctx.fillStyle = COLORS.line;
    ctx.lineWidth = 2.5 * k;
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 4 * k;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    for (const p of [a, b]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5 * k, 0, Math.PI * 2);
      ctx.fill();
    }
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const label = metres ? `${metres} m  ·  ${len.toFixed(0)} px` : `${len.toFixed(0)} px`;
    this.label(label, (a.x + b.x) / 2 + 10 * k, (a.y + b.y) / 2, COLORS.line, '#111');
    ctx.restore();
  }

  drawTracks(meter: SpeedMeter) {
    const ctx = this.ctx;
    const k = this.videoPerScreen();
    const ground = meter.cal.mode === 'ground';
    for (const tr of meter.tracks) {
      const color = COLORS[tr.status];
      // Trail of the measured reference point.
      if (tr.trail.length > 1) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.6;
        ctx.lineWidth = 2 * k;
        ctx.beginPath();
        tr.trail.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      const { x1, y1, x2, y2 } = tr.box;
      ctx.strokeStyle = color;
      ctx.lineWidth = (tr.id === meter.currentId ? 3 : 2) * k;
      ctx.setLineDash(tr.status === 'approaching' || tr.status === 'missed' ? [6 * k, 4 * k] : []);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      if (ground) {
        // The ground-contact point the trap measures.
        const g = groundPoint(tr.box);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(g.x, g.y, 3.5 * k, 0, Math.PI * 2);
        ctx.fill();
      }
      this.label(this.trackText(tr, meter), x1, y1 - 4 * k, color, 'rgba(10,12,18,0.85)', true);
    }
  }

  private trackText(tr: GunTrack, meter: SpeedMeter): string {
    const name = tr.classId >= 0 ? className(tr.classId) : 'target';
    const kmh = (v: number | null) => (v === null ? '' : `${Math.round(v)} km/h`);
    switch (tr.status) {
      case 'approaching':
        return `#${tr.id} ${name}`;
      case 'measuring':
        return `#${tr.id} ${kmh(tr.currentKmh) || 'measuring…'}`;
      case 'measured':
        return tr.fullCrossing ? `#${tr.id} ✓ ${kmh(tr.finalKmh)}` : `#${tr.id} ≈ ${kmh(tr.finalKmh)} (partial)`;
      case 'missed':
        return `#${tr.id} ${name} – not measured`;
      case 'tracking': {
        const mpp = meter.cal.metres / (Math.hypot(meter.cal.b.x - meter.cal.a.x, meter.cal.b.y - meter.cal.a.y) * tr.scale);
        return `${name} ${kmh(tr.currentKmh) || 'measuring…'}  (1 px = ${mpp.toFixed(3)} m)`;
      }
    }
  }

  /** Text with a background box. `above`: (x, y) is the bottom-left corner. */
  private label(text: string, x: number, y: number, color: string, bg: string, above = false) {
    const ctx = this.ctx;
    const k = this.videoPerScreen();
    const size = 14 * k;
    ctx.save();
    ctx.font = `600 ${size}px system-ui, -apple-system, Segoe UI, sans-serif`;
    const w = ctx.measureText(text).width + 10 * k;
    const h = size + 8 * k;
    const lx = Math.max(0, Math.min(x, this.canvas.width - w));
    let ly = above ? y - h : y - h / 2;
    if (ly < 0) ly = above ? y + h + 8 * k : 0;
    ctx.fillStyle = bg;
    ctx.fillRect(lx, ly, w, h);
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, 3 * k, h);
    ctx.fillStyle = color === COLORS.line ? COLORS.line : '#fff';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, lx + 6 * k, ly + h / 2 + 1 * k);
    ctx.restore();
  }
}

/** Classes the gun detects: road vehicles for a road calibration, anything that moves otherwise. */
export const GUN_CLASSES = {
  ground: CLASS_PRESETS.vehicles.classes,
  object: CLASS_PRESETS.moving.classes,
};
