export type ModelId = 'tiny' | 'small';

export interface ModelSpec {
  id: ModelId;
  file: string;
  inputSize: number;
  label: string;
  sizeMb: number;
}

/** YOLOX COCO models (Apache-2.0, Megvii). Files are fetched by scripts/fetch-model.mjs. */
export const MODELS: Record<ModelId, ModelSpec> = {
  tiny: { id: 'tiny', file: 'yolox_tiny.onnx', inputSize: 416, label: 'Fast · YOLOX-Tiny', sizeMb: 20 },
  small: { id: 'small', file: 'yolox_s.onnx', inputSize: 640, label: 'Accurate · YOLOX-S', sizeMb: 36 },
};
