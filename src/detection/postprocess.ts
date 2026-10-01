import type { Detection } from './types';

export const STRIDES = [8, 16, 32] as const;

/** Intersection-over-union of two boxes. */
export function iou(a: Detection, b: Detection): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

/**
 * Non-maximum suppression. Boxes of the same class are suppressed above `iouThreshold`;
 * near-identical boxes of different classes (the detector unsure whether a van is a
 * "car", "truck" or "bus") above `crossClassIou`. Returns detections sorted by score.
 */
export function nms(dets: Detection[], iouThreshold: number, crossClassIou = 0.6): Detection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const d of sorted) {
    const suppressed = kept.some((k) => {
      const o = iou(k, d);
      return k.classId === d.classId ? o > iouThreshold : o > crossClassIou;
    });
    if (!suppressed) kept.push(d);
  }
  return kept;
}

export interface DecodeOptions {
  inputSize: number;
  numClasses: number;
  /** Letterbox scale (model px = source px × ratio). */
  ratio: number;
  /** Top-left of the decoded region in the source frame (for tiles), source px. */
  offsetX?: number;
  offsetY?: number;
  /** Source frame size, used to clip boxes. */
  imageWidth: number;
  imageHeight: number;
  scoreThreshold: number;
  iouThreshold: number;
  /** Restrict to these classes; null = all classes. */
  classFilter: ReadonlySet<number> | null;
}

/**
 * Decodes raw YOLOX output `[N, 5 + C]` (cx, cy, w, h in grid units, obj, cls…; obj/cls
 * already sigmoided) into boxes in source-image coordinates, then applies NMS.
 */
export function decodeYolox(output: Float32Array, opts: DecodeOptions): Detection[] {
  const { inputSize, numClasses, ratio, imageWidth, imageHeight, scoreThreshold, classFilter } =
    opts;
  const ox = opts.offsetX ?? 0;
  const oy = opts.offsetY ?? 0;
  const stride = 5 + numClasses;
  const candidates: Detection[] = [];
  let row = 0;
  for (const s of STRIDES) {
    const g = Math.floor(inputSize / s);
    for (let gy = 0; gy < g; gy++) {
      for (let gx = 0; gx < g; gx++, row++) {
        const o = row * stride;
        if (o + stride > output.length) return nms(candidates, opts.iouThreshold);
        const obj = output[o + 4];
        if (obj < scoreThreshold) continue;
        let best = -1;
        let bestScore = 0;
        for (let c = 0; c < numClasses; c++) {
          if (classFilter && !classFilter.has(c)) continue;
          const sc = output[o + 5 + c];
          if (sc > bestScore) {
            bestScore = sc;
            best = c;
          }
        }
        const score = obj * bestScore;
        if (best < 0 || score < scoreThreshold) continue;
        const cx = (output[o] + gx) * s;
        const cy = (output[o + 1] + gy) * s;
        const w = Math.exp(output[o + 2]) * s;
        const h = Math.exp(output[o + 3]) * s;
        candidates.push({
          x1: clamp(ox + (cx - w / 2) / ratio, 0, imageWidth),
          y1: clamp(oy + (cy - h / 2) / ratio, 0, imageHeight),
          x2: clamp(ox + (cx + w / 2) / ratio, 0, imageWidth),
          y2: clamp(oy + (cy + h / 2) / ratio, 0, imageHeight),
          score,
          classId: best,
        });
      }
    }
  }
  return nms(candidates, opts.iouThreshold);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
