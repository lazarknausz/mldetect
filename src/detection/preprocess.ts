export const INPUT_SIZE = 416;
export const PAD_VALUE = 114;

/** Scale factor used to letterbox a frame into the square model input (top-left aligned). */
export function letterboxRatio(width: number, height: number, inputSize = INPUT_SIZE): number {
  return Math.min(inputSize / width, inputSize / height);
}

/**
 * Converts RGBA pixels of the letterboxed input image into the CHW float tensor YOLOX
 * expects: BGR channel order, raw 0–255 values (no normalisation).
 */
export function rgbaToBgrChw(
  rgba: Uint8ClampedArray | Uint8Array,
  size = INPUT_SIZE,
  out = new Float32Array(3 * size * size),
): Float32Array {
  const plane = size * size;
  for (let i = 0, p = 0; p < plane; i += 4, p++) {
    out[p] = rgba[i + 2]; // B
    out[plane + p] = rgba[i + 1]; // G
    out[2 * plane + p] = rgba[i]; // R
  }
  return out;
}
