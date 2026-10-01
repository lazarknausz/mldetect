import type { AircraftSize, ClassPreset } from './detection/classes';
import type { ModelId } from './detection/models';
import type { TileMode } from './detection/tiling';

export interface Settings {
  model: ModelId;
  preset: ClassPreset;
  /** Extra detector passes on zoomed-in tiles, for small / distant objects. */
  tiling: TileMode;
  /** Minimum detector confidence for starting a track (ByteTrack high threshold). */
  confidence: number;
  /** Camera zoom relative to a phone's main lens (sets the field of view for speeds). */
  zoom: number;
  /** What kind of aircraft "airplane" detections are (sets their size for speeds). */
  aircraft: AircraftSize;
  showTrails: boolean;
  showPredictions: boolean;
  showLabels: boolean;
  /** How far ahead the predicted path is drawn, seconds. */
  horizonSec: number;
  playbackRate: number;
  /** Frames per second sampled by precise analysis. */
  analysisFps: number;
}

export const DEFAULT_SETTINGS: Settings = {
  model: 'tiny',
  preset: 'moving',
  tiling: 'standard',
  confidence: 0.35,
  zoom: 1,
  aircraft: 'narrowbody',
  showTrails: true,
  showPredictions: true,
  showLabels: true,
  horizonSec: 1.5,
  playbackRate: 1,
  analysisFps: 15,
};
