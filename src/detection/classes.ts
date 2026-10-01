/** COCO-80 class names in YOLOX output order. */
export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
  'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
  'teddy bear', 'hair drier', 'toothbrush',
] as const;

export const NUM_CLASSES = COCO_CLASSES.length;

export const CLASS_ID = Object.fromEntries(COCO_CLASSES.map((n, i) => [n, i])) as Record<
  (typeof COCO_CLASSES)[number],
  number
>;

/**
 * Typical real-world size used to turn pixels into metres.
 * - `motion`: the object travels along its `length` axis; `cross` is its extent
 *   perpendicular to that (≈ height in side views, width in top-down views).
 * - `height`: upright objects (people) are scaled by their height.
 */
export type ReferenceSize =
  | { kind: 'motion'; length: number; cross: number }
  | { kind: 'height'; height: number };

export const REFERENCE_SIZE: Partial<Record<number, ReferenceSize>> = {
  [CLASS_ID.person]: { kind: 'height', height: 1.7 },
  [CLASS_ID.bicycle]: { kind: 'motion', length: 1.8, cross: 1.1 },
  [CLASS_ID.car]: { kind: 'motion', length: 4.5, cross: 1.7 },
  [CLASS_ID.motorcycle]: { kind: 'motion', length: 2.1, cross: 1.2 },
  [CLASS_ID.airplane]: { kind: 'motion', length: 38, cross: 12 },
  [CLASS_ID.bus]: { kind: 'motion', length: 12, cross: 3 },
  [CLASS_ID.train]: { kind: 'motion', length: 25, cross: 3.8 },
  // COCO "truck" covers vans and pickups as well as lorries.
  [CLASS_ID.truck]: { kind: 'motion', length: 7, cross: 2.8 },
  [CLASS_ID.boat]: { kind: 'motion', length: 10, cross: 3 },
  [CLASS_ID.bird]: { kind: 'motion', length: 0.4, cross: 0.6 },
  [CLASS_ID.horse]: { kind: 'motion', length: 2.4, cross: 1.6 },
  [CLASS_ID.dog]: { kind: 'motion', length: 0.9, cross: 0.6 },
};

/** Above this the estimate is almost certainly a tracking/scale error and is not shown. */
export const MAX_PLAUSIBLE_KMH: Partial<Record<number, number>> = {
  [CLASS_ID.person]: 45,
  [CLASS_ID.bicycle]: 90,
  [CLASS_ID.car]: 320,
  [CLASS_ID.motorcycle]: 330,
  [CLASS_ID.airplane]: 1100,
  [CLASS_ID.bus]: 200,
  [CLASS_ID.train]: 450,
  [CLASS_ID.truck]: 220,
  [CLASS_ID.boat]: 150,
  [CLASS_ID.bird]: 200,
  [CLASS_ID.horse]: 90,
  [CLASS_ID.dog]: 70,
};

export type ClassPreset = 'vehicles' | 'aircraft' | 'moving' | 'all';

export const CLASS_PRESETS: Record<ClassPreset, { label: string; classes: number[] | null }> = {
  vehicles: {
    label: 'Road vehicles',
    classes: [CLASS_ID.car, CLASS_ID.truck, CLASS_ID.bus, CLASS_ID.motorcycle, CLASS_ID.bicycle],
  },
  aircraft: { label: 'Aircraft & birds', classes: [CLASS_ID.airplane, CLASS_ID.bird, CLASS_ID.kite] },
  moving: {
    label: 'Anything that moves',
    classes: Object.keys(REFERENCE_SIZE).map(Number),
  },
  all: { label: 'All 80 COCO classes', classes: null },
};

export function className(classId: number): string {
  return COCO_CLASSES[classId] ?? `class ${classId}`;
}
