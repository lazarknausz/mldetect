import type { ClassPreset } from './detection/classes';
import type { ModelId } from './detection/models';

export interface Settings {
  model: ModelId;
  preset: ClassPreset;
  /** Minimum detector confidence for starting a track (ByteTrack high threshold). */
  confidence: number;
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
  confidence: 0.4,
  showTrails: true,
  showPredictions: true,
  showLabels: true,
  horizonSec: 1.5,
  playbackRate: 1,
  analysisFps: 15,
};
