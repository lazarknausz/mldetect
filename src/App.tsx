import { useCallback, useEffect, useRef, useState } from 'react';
import { ControlsPanel } from './components/ControlsPanel';
import { TrackTable } from './components/TrackTable';
import { VideoDropzone } from './components/VideoDropzone';
import { VideoStage } from './components/VideoStage';
import { DetectorClient } from './detection/detectorClient';
import { MODELS } from './detection/models';
import type { DetectorStatus } from './detection/types';
import { framesToCsv } from './engine/csv';
import { TrackingEngine, type EngineMode, type EngineStats } from './engine/TrackingEngine';
import { DEFAULT_SETTINGS, type Settings } from './settings';
import type { TrackSnapshot } from './tracking/types';

const TABLE_REFRESH_MS = 150;
const EMPTY_STATS: EngineStats = { fps: 0, inferMs: 0, totalObjects: 0, passes: 1, camera: 'static' };

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<TrackingEngine | null>(null);

  const [status, setStatus] = useState<DetectorStatus>({ state: 'idle' });
  const [detector, setDetector] = useState<DetectorClient | null>(null);
  const [video, setVideo] = useState<{ url: string; name: string } | null>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [tracks, setTracks] = useState<TrackSnapshot[]>([]);
  const [stats, setStats] = useState<EngineStats>(EMPTY_STATS);
  const [mode, setMode] = useState<EngineMode>('live');
  const [progress, setProgress] = useState(0);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [canExport, setCanExport] = useState(false);

  // Load the model as soon as the page opens (and again if the user switches model).
  useEffect(() => {
    const spec = MODELS[settings.model];
    const d = new DetectorClient(spec, `${import.meta.env.BASE_URL}models/${spec.file}`, setStatus);
    setDetector(d);
    return () => d.terminate();
  }, [settings.model]);

  // Throttle table updates; the canvas overlay itself renders every animation frame.
  const pending = useRef<{ tracks: TrackSnapshot[]; stats: EngineStats } | null>(null);
  const timer = useRef<number | null>(null);
  const pushTracks = useCallback((t: TrackSnapshot[], s: EngineStats) => {
    pending.current = { tracks: t, stats: s };
    if (timer.current !== null) return;
    timer.current = window.setTimeout(() => {
      timer.current = null;
      if (pending.current) {
        setTracks(pending.current.tracks);
        setStats(pending.current.stats);
      }
    }, TABLE_REFRESH_MS);
  }, []);

  // Create the engine once both the model and the video are ready.
  const modelReady = status.state === 'ready';
  useEffect(() => {
    if (!modelReady || !videoLoaded || !videoRef.current || !canvasRef.current || !detector) return;
    const engine = new TrackingEngine(videoRef.current, canvasRef.current, detector, settings, {
      onTracks: pushTracks,
      onMode: (m) => {
        setMode(m);
        if (m === 'replay') setCanExport(true);
      },
      onProgress: setProgress,
      onError: setError,
    });
    engineRef.current = engine;
    engine.goLive();
    return () => {
      engine.dispose();
      engineRef.current = null;
    };
    // Settings are pushed separately below; recreating the engine would lose tracks.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelReady, videoLoaded, video, detector, pushTracks]);

  useEffect(() => engineRef.current?.updateSettings(settings), [settings]);
  useEffect(() => engineRef.current?.select(selectedId), [selectedId]);

  const loadFile = (file: File) => {
    if (video) URL.revokeObjectURL(video.url);
    setVideoLoaded(false);
    setTracks([]);
    setStats(EMPTY_STATS);
    setSelectedId(null);
    setCanExport(false);
    setError(null);
    setMode('live');
    setVideo({ url: URL.createObjectURL(file), name: file.name });
  };

  const exportCsv = () => {
    const frames = engineRef.current?.analysisFrames;
    if (!frames?.length) return;
    const blob = new Blob([framesToCsv(frames)], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${(video?.name ?? 'video').replace(/\.[^.]+$/, '')}-tracks.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  return (
    <div className="app">
      <header>
        <div className="brand">
          <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
            <rect x="3" y="6" width="11" height="8" rx="1.5" fill="none" stroke="currentColor" strokeWidth="2" />
            <path d="M14 10h7m-3-3 3 3-3 3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <h1>MLDetect</h1>
        </div>
        <p>Track every car, plane, boat or bird in a video: speed, direction and where it is heading next.</p>
        <a className="nav-link" href="speed-gun.html">
          Speed gun (calibrated) →
        </a>
      </header>

      {error && (
        <div className="error" role="alert">
          {error}
          <button className="ghost" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <main>
        <div className="left">
          {video ? (
            <VideoStage
              src={video.url}
              videoRef={videoRef}
              canvasRef={canvasRef}
              disabled={mode === 'analyzing'}
              onLoaded={() => setVideoLoaded(true)}
              onStageClick={(x, y) => setSelectedId(engineRef.current?.pick(x, y) ?? null)}
            />
          ) : (
            <VideoDropzone onFile={loadFile} />
          )}
          {video && !modelReady && status.state !== 'error' && (
            <p className="hint">Waiting for the detection model to load…</p>
          )}
          <TrackTable tracks={tracks} selectedId={selectedId} onSelect={setSelectedId} />
        </div>
        <ControlsPanel
          settings={settings}
          onChange={setSettings}
          status={status}
          stats={stats}
          inView={tracks.filter((t) => t.state !== 'lost').length}
          mode={mode}
          progress={progress}
          hasVideo={videoLoaded}
          canExport={canExport}
          onAnalyze={() => void engineRef.current?.analyze()}
          onLive={() => engineRef.current?.goLive()}
          onExport={exportCsv}
          onNewVideo={() => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'video/*';
            input.onchange = () => input.files?.[0] && loadFile(input.files[0]);
            input.click();
          }}
        />
      </main>
      <footer>
        Detection: {MODELS[settings.model].label.split(' · ')[1]} (Apache-2.0) via ONNX Runtime Web ·
        Tracking: ByteTrack-style Kalman + IoU association with camera-motion compensation · Runs 100%
        locally in your browser.
      </footer>
    </div>
  );
}
