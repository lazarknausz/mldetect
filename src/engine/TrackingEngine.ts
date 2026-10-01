import { AIRCRAFT_SIZES, CLASS_ID, CLASS_PRESETS } from '../detection/classes';
import type { DetectorClient } from '../detection/detectorClient';
import { MOTION_IMAGE_SIZE } from '../detection/protocol';
import { planRegions, regionRatio } from '../detection/tiling';
import { computeView, hitTest, renderOverlay } from '../render/overlay';
import type { Settings } from '../settings';
import { Tracker } from '../tracking/tracker';
import type { CameraState, TrackSnapshot } from '../tracking/types';

export type EngineMode = 'live' | 'analyzing' | 'replay';

export interface FrameResult {
  t: number;
  tracks: TrackSnapshot[];
}

export interface EngineStats {
  /** Detection frames processed per second (live mode). */
  fps: number;
  inferMs: number;
  totalObjects: number;
  /** Detector passes per frame (full frame + tiles). */
  passes: number;
  camera: CameraState;
}

export interface EngineCallbacks {
  onTracks(tracks: TrackSnapshot[], stats: EngineStats): void;
  onMode(mode: EngineMode): void;
  onProgress(fraction: number): void;
  onError(message: string): void;
}

/** Lowest detector score kept at all (ByteTrack uses these to extend existing tracks). */
const LOW_SCORE = 0.1;

/**
 * Glue between the <video>, the detector worker, the tracker and the overlay canvas.
 * Not a React component: it runs its own frame callbacks and render loop.
 */
export class TrackingEngine {
  private tracker = new Tracker();
  private settings: Settings;
  private mode: EngineMode = 'live';
  private latest: FrameResult = { t: 0, tracks: [] };
  private frames: FrameResult[] = [];
  private busy = false;
  private disposed = false;
  private cancelAnalysis = false;
  private rafHandle = 0;
  private rvfcHandle = 0;
  private selectedId: number | null = null;
  private stats: EngineStats = { fps: 0, inferMs: 0, totalObjects: 0, passes: 1, camera: 'static' };
  /** Changes whenever tracking restarts, so camera motion is not measured across a jump. */
  private sequence = 0;
  private processedTimes: number[] = [];
  private lastEmitted: FrameResult | null = null;

  constructor(
    private video: HTMLVideoElement,
    private canvas: HTMLCanvasElement,
    private detector: DetectorClient,
    settings: Settings,
    private cb: EngineCallbacks,
  ) {
    this.settings = settings;
    this.applyTrackerSettings();
    this.video.playbackRate = settings.playbackRate;
    this.rvfcHandle = this.video.requestVideoFrameCallback(this.onVideoFrame);
    this.video.addEventListener('seeked', this.onSeeked);
    this.rafHandle = requestAnimationFrame(this.render);
  }

  dispose(): void {
    this.disposed = true;
    this.cancelAnalysis = true;
    cancelAnimationFrame(this.rafHandle);
    this.video.cancelVideoFrameCallback(this.rvfcHandle);
    this.video.removeEventListener('seeked', this.onSeeked);
  }

  updateSettings(next: Settings): void {
    const prev = this.settings;
    this.settings = next;
    this.applyTrackerSettings();
    this.video.playbackRate = next.playbackRate;
    if ((prev.preset !== next.preset || prev.tiling !== next.tiling) && this.mode === 'live') this.resetTracking();
  }

  select(id: number | null): void {
    this.selectedId = id;
  }

  /** Returns the id of the track under a point given in stage CSS pixels. */
  pick(stageX: number, stageY: number): number | null {
    const view = this.view();
    if (!view) return null;
    return hitTest(this.currentTracks().tracks, (stageX - view.ox) / view.scale, (stageY - view.oy) / view.scale);
  }

  get analysisFrames(): readonly FrameResult[] {
    return this.frames;
  }

  resetTracking(): void {
    this.restartTracker();
    this.latest = { t: this.video.currentTime, tracks: [] };
    this.processedTimes = [];
    this.emit(this.latest);
  }

  /** Switches back to live detection (discarding nothing: the analysis stays exportable). */
  goLive(): void {
    this.cancelAnalysis = true;
    this.setMode('live');
    this.resetTracking();
    if (this.video.paused) void this.processFrame(this.video.currentTime);
  }

  /** Steps through the whole video at a fixed rate, processing every step. */
  async analyze(): Promise<void> {
    if (this.mode === 'analyzing') return;
    const video = this.video;
    video.pause();
    this.setMode('analyzing');
    this.cancelAnalysis = false;
    // Wait for any in-flight live detection to finish.
    while (this.busy) await sleep(10);
    this.restartTracker();
    this.frames = [];
    const step = 1 / this.settings.analysisFps;
    const duration = video.duration;
    try {
      for (let target = 0; target <= duration; target += step) {
        if (this.cancelAnalysis || this.disposed) return;
        const t = await seekTo(video, Math.min(target, Math.max(0, duration - 0.001)));
        this.cb.onProgress(Math.min(1, (target + step) / duration));
        // Videos with fewer fps than the analysis rate repeat frames; process each once.
        const prev = this.frames[this.frames.length - 1];
        if (prev && Math.abs(prev.t - t) < 1e-4) continue;
        const tracks = await this.detectAndTrack(t);
        const frame = { t, tracks };
        this.frames.push(frame);
        this.latest = frame;
        this.emit(frame);
      }
    } catch (err) {
      this.cb.onError(`Analysis failed: ${String((err as Error)?.message ?? err)}`);
      this.setMode('live');
      return;
    }
    this.cb.onProgress(1);
    this.setMode('replay');
    await seekTo(video, 0);
  }

  // ---------------------------------------------------------------- internals

  private applyTrackerSettings() {
    const { opts } = this.tracker;
    opts.highThreshold = this.settings.confidence;
    opts.zoom = this.settings.zoom;
    opts.referenceSizes = { [CLASS_ID.airplane]: AIRCRAFT_SIZES[this.settings.aircraft].ref };
  }

  private restartTracker() {
    this.tracker.reset();
    this.sequence++;
  }

  private setMode(mode: EngineMode) {
    this.mode = mode;
    this.cb.onMode(mode);
  }

  private onVideoFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
    if (this.disposed) return;
    this.rvfcHandle = this.video.requestVideoFrameCallback(this.onVideoFrame);
    if (this.mode === 'live' && !this.busy) void this.processFrame(meta.mediaTime);
  };

  private onSeeked = () => {
    // A user seek invalidates motion history; start fresh at the new position.
    if (this.mode === 'live') {
      this.restartTracker();
      this.latest = { t: this.video.currentTime, tracks: [] };
      this.emit(this.latest);
    }
  };

  private async processFrame(t: number) {
    if (this.busy) return;
    const last = this.tracker.lastTime;
    if (last !== null && Math.abs(t - last) < 1e-4) return;
    this.busy = true;
    try {
      if (last !== null && (t < last || t - last > 2)) this.restartTracker();
      const tracks = await this.detectAndTrack(t);
      if (this.mode !== 'live' || this.disposed) return;
      this.latest = { t, tracks };
      const now = performance.now();
      this.processedTimes.push(now);
      while (this.processedTimes.length && now - this.processedTimes[0] > 2000) this.processedTimes.shift();
      this.stats.fps = this.processedTimes.length / 2;
      this.emit(this.latest);
    } catch (err) {
      this.cb.onError(String((err as Error)?.message ?? err));
    } finally {
      this.busy = false;
    }
  }

  private async detectAndTrack(t: number): Promise<TrackSnapshot[]> {
    const video = this.video;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const inputSize = this.detector.model.inputSize;
    const regions = planRegions(vw, vh, inputSize, this.settings.tiling);
    // Resize once, to the finest resolution any region (or the motion estimate) needs.
    const bitmapScale = Math.min(
      1,
      Math.max(MOTION_IMAGE_SIZE / Math.max(vw, vh), ...regions.map((r) => regionRatio(r, inputSize))),
    );
    const bitmap = await createImageBitmap(video, {
      resizeWidth: Math.max(1, Math.round(vw * bitmapScale)),
      resizeHeight: Math.max(1, Math.round(vh * bitmapScale)),
      resizeQuality: 'medium',
    });
    const { detections, inferMs, camera } = await this.detector.detect(bitmap, {
      bitmapScale,
      imageWidth: vw,
      imageHeight: vh,
      regions,
      scoreThreshold: LOW_SCORE,
      classes: CLASS_PRESETS[this.settings.preset].classes,
      motion: { sequence: this.sequence, t },
    });
    this.stats.inferMs = inferMs;
    this.stats.passes = regions.length;
    const tracks = this.tracker.update(detections, t, vw, vh, camera);
    this.stats.totalObjects = this.tracker.totalConfirmed;
    this.stats.camera = this.tracker.cameraState;
    return tracks;
  }

  private emit(frame: FrameResult) {
    if (frame === this.lastEmitted) return;
    this.lastEmitted = frame;
    this.cb.onTracks(frame.tracks, { ...this.stats });
  }

  /** Tracks to show for the frame currently on screen. */
  private currentTracks(): FrameResult {
    if (this.mode !== 'replay' || !this.frames.length) return this.latest;
    const t = this.video.currentTime;
    // Binary search for the last analysed frame at or before t.
    let lo = 0;
    let hi = this.frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.frames[mid].t <= t + 1e-3) lo = mid;
      else hi = mid - 1;
    }
    return this.frames[lo];
  }

  private view() {
    const { videoWidth, videoHeight } = this.video;
    if (!videoWidth || !videoHeight) return null;
    const rect = this.video.getBoundingClientRect();
    return computeView(rect.width, rect.height, videoWidth, videoHeight);
  }

  private render = () => {
    if (this.disposed) return;
    this.rafHandle = requestAnimationFrame(this.render);
    const canvas = this.canvas;
    const rect = this.video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(rect.width * dpr);
    const ch = Math.round(rect.height * dpr);
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const view = this.view();
    if (!view) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const frame = this.currentTracks();
    if (this.mode === 'replay') this.emit(frame);
    // Extrapolate to the frame on screen; while analysing the overlay matches exactly.
    const dt = this.mode === 'analyzing' ? 0 : this.video.currentTime - frame.t;
    renderOverlay(ctx, frame.tracks, dt > 0 && dt < 0.5 ? dt : 0, view, {
      showTrails: this.settings.showTrails,
      showPredictions: this.settings.showPredictions,
      showLabels: this.settings.showLabels,
      horizonSec: this.settings.horizonSec,
      selectedId: this.selectedId,
    });
  };
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Seeks and resolves with the media time of the frame actually presented (falls back
 * to the requested time if the browser does not report a new frame).
 */
function seekTo(video: HTMLVideoElement, time: number): Promise<number> {
  return new Promise((resolve) => {
    let done = false;
    let handle = 0;
    const finish = (t: number) => {
      if (done) return;
      done = true;
      video.cancelVideoFrameCallback(handle);
      video.removeEventListener('seeked', onSeeked);
      resolve(t);
    };
    const onSeeked = () => setTimeout(() => finish(video.currentTime), 250);
    handle = video.requestVideoFrameCallback((_n, meta) => finish(meta.mediaTime));
    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
}
