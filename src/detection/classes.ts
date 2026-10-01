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
 * - `motion`: a block that travels along its `length` axis, `width` across (wingspan
 *   for aircraft and birds) and `height` tall. Which of these the bounding box shows
 *   depends on the viewpoint, which the speed estimator works out.
 * - `height`: upright objects (people) are scaled by their height.
 */
export type ReferenceSize =
  | { kind: 'motion'; length: number; width: number; height: number }
  | { kind: 'height'; height: number };

function block(length: number, width: number, height: number): ReferenceSize {
  return { kind: 'motion', length, width, height };
}

export const REFERENCE_SIZE: Partial<Record<number, ReferenceSize>> = {
  [CLASS_ID.person]: { kind: 'height', height: 1.7 },
  [CLASS_ID.bicycle]: block(1.8, 0.6, 1.1),
  [CLASS_ID.car]: block(4.5, 1.8, 1.5),
  [CLASS_ID.motorcycle]: block(2.1, 0.8, 1.2),
  [CLASS_ID.airplane]: block(38, 35, 12),
  [CLASS_ID.bus]: block(12, 2.55, 3.2),
  [CLASS_ID.train]: block(25, 3, 4.2),
  // COCO "truck" covers vans and pickups as well as lorries.
  [CLASS_ID.truck]: block(7, 2.4, 3),
  [CLASS_ID.boat]: block(10, 3.5, 3),
  [CLASS_ID.bird]: block(0.4, 1, 0.2),
  [CLASS_ID.horse]: block(2.4, 0.6, 1.7),
  [CLASS_ID.dog]: block(0.9, 0.3, 0.6),
};

export type AircraftSize = 'light' | 'regional' | 'narrowbody' | 'widebody';

/** COCO has a single "airplane" class, so the user can say what kind of aircraft it is. */
export const AIRCRAFT_SIZES: Record<AircraftSize, { label: string; ref: ReferenceSize }> = {
  light: { label: 'Light aircraft (≈ 8 m, e.g. Cessna)', ref: block(8.3, 11, 2.7) },
  regional: { label: 'Regional / business jet (≈ 28 m)', ref: block(28, 27, 7.5) },
  narrowbody: { label: 'Airliner, narrow-body (≈ 38 m, A320 / 737)', ref: block(38, 35, 12) },
  widebody: { label: 'Airliner, wide-body (≈ 65 m, 777 / A350)', ref: block(65, 63, 18.5) },
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

/** Classes the detector often confuses with each other (treated as one object when matching). */
export const CLASS_GROUPS: number[][] = [
  [CLASS_ID.car, CLASS_ID.truck, CLASS_ID.bus, CLASS_ID.train],
  [CLASS_ID.bicycle, CLASS_ID.motorcycle],
  [CLASS_ID.airplane, CLASS_ID.bird, CLASS_ID.kite],
  [CLASS_ID.boat, CLASS_ID.surfboard],
];
const GROUP_OF = new Map<number, number>();
CLASS_GROUPS.forEach((g, i) => g.forEach((c) => GROUP_OF.set(c, i)));

export function sameClassGroup(a: number, b: number): boolean {
  if (a === b) return true;
  const ga = GROUP_OF.get(a);
  return ga !== undefined && ga === GROUP_OF.get(b);
}

export function className(classId: number): string {
  return COCO_CLASSES[classId] ?? `class ${classId}`;
}
