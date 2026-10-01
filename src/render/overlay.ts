import { predictPath } from '../tracking/motion';
import type { TrackSnapshot } from '../tracking/types';

/** Maps source-video pixels to CSS pixels of the stage (object-fit: contain). */
export interface ViewTransform {
  scale: number;
  ox: number;
  oy: number;
}

export interface OverlayOptions {
  showTrails: boolean;
  showPredictions: boolean;
  showLabels: boolean;
  horizonSec: number;
  selectedId: number | null;
}

export function computeView(
  stageWidth: number,
  stageHeight: number,
  videoWidth: number,
  videoHeight: number,
): ViewTransform {
  const scale = Math.min(stageWidth / videoWidth, stageHeight / videoHeight);
  return {
    scale,
    ox: (stageWidth - videoWidth * scale) / 2,
    oy: (stageHeight - videoHeight * scale) / 2,
  };
}

export function trackColor(id: number, alpha = 1): string {
  const hue = (id * 137.508) % 360;
  return `hsla(${hue.toFixed(0)}, 90%, 60%, ${alpha})`;
}

export function formatSpeed(s: TrackSnapshot): string {
  if (s.measuring) return 'measuring…';
  if (s.stationary) return 'stationary';
  if (s.speedKmh !== null) return `≈ ${Math.round(s.speedKmh)} km/h`;
  return `${Math.round(s.speedPx)} px/s`;
}

/**
 * Draws all tracks. `dt` (seconds) extrapolates each track from its snapshot time to
 * the frame currently on screen, so boxes stay glued to objects even when detection
 * runs slower than playback.
 */
export function renderOverlay(
  ctx: CanvasRenderingContext2D,
  tracks: readonly TrackSnapshot[],
  dt: number,
  view: ViewTransform,
  opts: OverlayOptions,
): void {
  const X = (x: number) => view.ox + x * view.scale;
  const Y = (y: number) => view.oy + y * view.scale;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  // Selected track last, so it is drawn on top.
  const ordered = [...tracks].sort(
    (a, b) => Number(a.id === opts.selectedId) - Number(b.id === opts.selectedId),
  );

  for (const s of ordered) {
    const selected = s.id === opts.selectedId;
    const lost = s.state === 'lost';
    const dim = opts.selectedId !== null && !selected ? 0.45 : 1;
    const color = trackColor(s.id, (lost ? 0.6 : 1) * dim);
    const lead = Math.max(0, Math.min(dt, 0.5));
    const cx = s.cx + s.vx * lead;
    const cy = s.cy + s.vy * lead;
    const w = Math.max(2, s.w + s.vw * lead);
    const h = Math.max(2, s.h + s.vh * lead);

    // Trail of past positions, fading out.
    if (opts.showTrails && s.trail.length > 1) {
      ctx.lineWidth = selected ? 3 : 2;
      for (let i = 1; i < s.trail.length; i++) {
        ctx.strokeStyle = trackColor(s.id, (i / s.trail.length) * 0.8 * dim);
        ctx.beginPath();
        ctx.moveTo(X(s.trail[i - 1].x), Y(s.trail[i - 1].y));
        ctx.lineTo(X(s.trail[i].x), Y(s.trail[i].y));
        ctx.stroke();
      }
    }

    // Predicted path + ghost box at the horizon.
    if (opts.showPredictions && !s.stationary && !s.measuring) {
      const path = predictPath(cx, cy, s.vx, s.vy, s.turnRate, opts.horizonSec);
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = selected ? 2.5 : 1.75;
      ctx.strokeStyle = trackColor(s.id, 0.85 * dim);
      ctx.beginPath();
      path.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = trackColor(s.id, 0.9 * dim);
      for (const p of path) {
        if (p.t > 0 && Math.abs(p.t / 0.5 - Math.round(p.t / 0.5)) < 1e-6) {
          ctx.beginPath();
          ctx.arc(X(p.x), Y(p.y), 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      const end = path[path.length - 1];
      const gw = Math.max(2, w + s.vw * opts.horizonSec) * view.scale;
      const gh = Math.max(2, h + s.vh * opts.horizonSec) * view.scale;
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = trackColor(s.id, 0.5 * dim);
      ctx.strokeRect(X(end.x) - gw / 2, Y(end.y) - gh / 2, gw, gh);
      ctx.restore();
    }

    // Bounding box.
    const bx = X(cx - w / 2);
    const by = Y(cy - h / 2);
    const bw = w * view.scale;
    const bh = h * view.scale;
    ctx.save();
    if (lost) ctx.setLineDash([5, 4]);
    if (selected) {
      ctx.shadowColor = trackColor(s.id, 0.9);
      ctx.shadowBlur = 12;
    }
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(bx, by, bw, bh);
    ctx.restore();

    // Direction arrow from the box centre.
    if (!s.stationary && !s.measuring && s.speedPx > 0) {
      const len = Math.max(18, Math.min(90, s.speedPx * view.scale * 0.35));
      const ux = s.vx / s.speedPx;
      const uy = s.vy / s.speedPx;
      drawArrow(ctx, X(cx), Y(cy), X(cx) + ux * len, Y(cy) + uy * len, color, selected ? 3 : 2.25);
    }

    if (opts.showLabels) drawLabel(ctx, s, bx, by, bw, color, lost);
  }
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
  width: number,
) {
  const ang = Math.atan2(y1 - y0, x1 - x0);
  const head = 7 + width;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - Math.cos(ang) * head * 0.6, y1 - Math.sin(ang) * head * 0.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
  ctx.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x0, y0, width + 0.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  s: TrackSnapshot,
  bx: number,
  by: number,
  bw: number,
  color: string,
  lost: boolean,
) {
  const line1 = `#${s.id} ${s.label}${lost ? ' (lost)' : ''}`;
  const line2 =
    s.headingDeg === null
      ? formatSpeed(s)
      : `${formatSpeed(s)} · ${s.approach ?? `${s.compass} ${Math.round(s.headingDeg)}°`}`;
  ctx.save();
  ctx.font = '600 12px system-ui, -apple-system, Segoe UI, sans-serif';
  const pad = 4;
  const lh = 14;
  const tw = Math.max(ctx.measureText(line1).width, ctx.measureText(line2).width) + pad * 2;
  const th = lh * 2 + pad;
  const canvasW = ctx.canvas.clientWidth || ctx.canvas.width;
  const lx = Math.max(0, Math.min(bx, canvasW - tw));
  const ly = by - th - 2 >= 0 ? by - th - 2 : by + 2;
  ctx.fillStyle = 'rgba(10, 12, 18, 0.78)';
  ctx.beginPath();
  ctx.roundRect(lx, ly, Math.max(tw, Math.min(bw, tw)), th, 4);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillRect(lx, ly, 3, th);
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'top';
  ctx.fillText(line1, lx + pad + 2, ly + pad / 2 + 1);
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.font = '500 12px system-ui, -apple-system, Segoe UI, sans-serif';
  ctx.fillText(line2, lx + pad + 2, ly + pad / 2 + 1 + lh);
  ctx.restore();
}

/** Returns the id of the track whose box contains the point (source-video px). */
export function hitTest(tracks: readonly TrackSnapshot[], x: number, y: number): number | null {
  let best: TrackSnapshot | null = null;
  for (const s of tracks) {
    if (Math.abs(x - s.cx) <= s.w / 2 && Math.abs(y - s.cy) <= s.h / 2) {
      if (!best || s.w * s.h < best.w * best.h) best = s;
    }
  }
  return best?.id ?? null;
}
