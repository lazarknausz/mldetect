import { sameClassGroup } from './classes';
import { nms } from './postprocess';
import type { Detection } from './types';

/**
 * Sliced inference for small / distant objects. The detector sees a fixed-size square
 * (416 or 640 px), so a 1080p frame is shrunk ~4.6× and a car 40 px long becomes ~9 px:
 * too small to detect. Running the model again on overlapping crops ("tiles") of the
 * frame shows those objects at 2–3× the size; the full frame is still processed for
 * objects too big to fit inside one tile.
 */
export type TileMode = 'off' | 'standard' | 'max';

export const TILE_MODES: Record<TileMode, { label: string; gain: number }> = {
  off: { label: 'Off (fastest)', gain: 1 },
  standard: { label: 'On · 2–4 tiles', gain: 1.7 },
  max: { label: 'Maximum · 6–9 tiles (slow)', gain: 3 },
};

/** A crop of the source frame, in source pixels. */
export interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Minimum overlap between neighbouring tiles, as a fraction of the tile size. */
const MIN_OVERLAP = 0.18;

function spans(length: number, tile: number): number[] {
  if (tile >= length) return [0];
  const overlap = MIN_OVERLAP * tile;
  const n = Math.ceil((length - overlap) / (tile - overlap));
  const step = (length - tile) / (n - 1);
  return Array.from({ length: n }, (_, i) => Math.round(i * step));
}

/**
 * Regions to run the detector on: the full frame first, then the tiles. Tiles are
 * skipped when they would not magnify the image noticeably (small videos).
 */
export function planRegions(width: number, height: number, inputSize: number, mode: TileMode): Region[] {
  const full: Region = { x: 0, y: 0, w: width, h: height };
  const gain = TILE_MODES[mode].gain;
  if (gain <= 1) return [full];
  const long = Math.max(width, height);
  // Never zoom in beyond 1:1 (upscaling adds no detail).
  const side = Math.max(inputSize, long / gain);
  if (long / side < 1.25) return [full];
  const tw = Math.min(width, Math.round(side));
  const th = Math.min(height, Math.round(side));
  const out = [full];
  for (const y of spans(height, th)) for (const x of spans(width, tw)) out.push({ x, y, w: tw, h: th });
  return out;
}

/** Model-input pixels per source pixel for a region (the region is letterboxed into the input). */
export function regionRatio(r: Region, inputSize: number): number {
  return Math.min(inputSize / r.w, inputSize / r.h);
}

/**
 * Drops tile detections cut off by an inner tile border: the object continues in the
 * neighbouring tile (or is large enough for the full-frame pass), and a truncated box
 * would otherwise show up as a second, smaller object.
 */
export function dropCutAtTileEdge(
  dets: Detection[],
  tile: Region,
  frameWidth: number,
  frameHeight: number,
  margin = 2,
): Detection[] {
  const left = tile.x > 0 ? tile.x + margin : -Infinity;
  const top = tile.y > 0 ? tile.y + margin : -Infinity;
  const right = tile.x + tile.w < frameWidth ? tile.x + tile.w - margin : Infinity;
  const bottom = tile.y + tile.h < frameHeight ? tile.y + tile.h - margin : Infinity;
  return dets.filter((d) => d.x1 > left && d.y1 > top && d.x2 < right && d.y2 < bottom);
}

function intersectionOverSmaller(a: Detection, b: Detection): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const smaller = Math.min((a.x2 - a.x1) * (a.y2 - a.y1), (b.x2 - b.x1) * (b.y2 - b.y1));
  return smaller > 0 ? (ix * iy) / smaller : 0;
}

/**
 * Merges full-frame and tile detections. A tile box lying inside a confident full-frame
 * box of the same kind of object is a part of it (e.g. the cab of a truck) and is
 * dropped. The same object found in several overlapping views is then kept once
 * (highest score), as in ordinary NMS.
 */
export function mergeRegionDetections(
  full: Detection[],
  tiles: Detection[],
  iouThreshold: number,
  crossClassIou = 0.6,
): Detection[] {
  const area = (d: Detection) => (d.x2 - d.x1) * (d.y2 - d.y1);
  const wholeObjects = tiles.filter(
    (t) =>
      !full.some(
        (f) =>
          f.score >= 0.3 &&
          sameClassGroup(f.classId, t.classId) &&
          area(t) < 0.8 * area(f) &&
          intersectionOverSmaller(f, t) > 0.75,
      ),
  );
  return nms([...full, ...wholeObjects], iouThreshold, crossClassIou);
}
