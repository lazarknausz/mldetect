import { CLASS_PRESETS, type ClassPreset } from '../detection/classes';
import { MODELS, type ModelId } from '../detection/models';
import type { DetectorStatus } from '../detection/types';
import type { EngineMode, EngineStats } from '../engine/TrackingEngine';
import type { Settings } from '../settings';

interface Props {
  settings: Settings;
  onChange(s: Settings): void;
  status: DetectorStatus;
  stats: EngineStats;
  inView: number;
  mode: EngineMode;
  progress: number;
  hasVideo: boolean;
  canExport: boolean;
  onAnalyze(): void;
  onLive(): void;
  onExport(): void;
  onNewVideo(): void;
}

export function ControlsPanel(p: Props) {
  const { settings: s, onChange } = p;
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => onChange({ ...s, [k]: v });
  const ready = p.status.state === 'ready';

  return (
    <section className="panel">
      <div className="status-row">
        <span className={`dot ${p.status.state}`} />
        <span>
          {p.status.state === 'ready'
            ? `Model ready · ${p.status.backend === 'webgpu' ? 'WebGPU' : 'WASM (CPU)'}`
            : p.status.state === 'error'
              ? p.status.message
              : (p.status.message ?? 'Loading…')}
        </span>
      </div>

      <div className="stats">
        <div><b>{p.inView}</b><span>in view</span></div>
        <div><b>{p.stats.totalObjects}</b><span>total seen</span></div>
        <div><b>{p.mode === 'live' ? p.stats.fps.toFixed(1) : '–'}</b><span>detections/s</span></div>
        <div><b>{p.stats.inferMs ? Math.round(p.stats.inferMs) : '–'}</b><span>ms / frame</span></div>
      </div>

      <h3>Mode</h3>
      {p.mode === 'analyzing' ? (
        <div className="progress-block">
          <div className="progress"><div style={{ width: `${(p.progress * 100).toFixed(1)}%` }} /></div>
          <div className="row">
            <span>Analysing every frame… {Math.round(p.progress * 100)}%</span>
            <button onClick={p.onLive}>Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <p className="hint">
            {p.mode === 'live'
              ? 'Live: press play. Frames are analysed as fast as your device allows.'
              : 'Replay: showing the precise analysis. Play or scrub freely.'}
          </p>
          <div className="row wrap">
            <button className="primary" disabled={!ready || !p.hasVideo} onClick={p.onAnalyze}>
              Precise analysis
            </button>
            <select
              aria-label="Analysis frame rate"
              value={s.analysisFps}
              onChange={(e) => set('analysisFps', Number(e.target.value))}
            >
              {[5, 10, 15, 30].map((f) => <option key={f} value={f}>{f} fps</option>)}
            </select>
            {p.mode === 'replay' && <button onClick={p.onLive}>Back to live</button>}
            <button disabled={!p.canExport} onClick={p.onExport}>Export CSV</button>
          </div>
        </>
      )}

      <h3>Detect</h3>
      <label className="field" title="Accurate finds smaller / more distant objects but is about 3× slower.">
        <span>Model</span>
        <select value={s.model} onChange={(e) => set('model', e.target.value as ModelId)} disabled={p.mode === 'analyzing'}>
          {Object.values(MODELS).map((m) => (
            <option key={m.id} value={m.id}>{m.label} ({m.sizeMb} MB)</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Objects</span>
        <select value={s.preset} onChange={(e) => set('preset', e.target.value as ClassPreset)}>
          {(Object.keys(CLASS_PRESETS) as ClassPreset[]).map((k) => (
            <option key={k} value={k}>{CLASS_PRESETS[k].label}</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Min. confidence <b>{Math.round(s.confidence * 100)}%</b></span>
        <input
          type="range" min={0.2} max={0.8} step={0.05} value={s.confidence}
          onChange={(e) => set('confidence', Number(e.target.value))}
        />
      </label>

      <h3>Display</h3>
      <div className="toggles">
        <label><input type="checkbox" checked={s.showLabels} onChange={(e) => set('showLabels', e.target.checked)} /> Labels</label>
        <label><input type="checkbox" checked={s.showTrails} onChange={(e) => set('showTrails', e.target.checked)} /> Trails</label>
        <label><input type="checkbox" checked={s.showPredictions} onChange={(e) => set('showPredictions', e.target.checked)} /> Predicted path</label>
      </div>
      <label className="field">
        <span>Prediction horizon <b>{s.horizonSec} s</b></span>
        <input
          type="range" min={0.5} max={4} step={0.5} value={s.horizonSec}
          onChange={(e) => set('horizonSec', Number(e.target.value))}
        />
      </label>
      <label className="field">
        <span>Playback speed</span>
        <select value={s.playbackRate} onChange={(e) => set('playbackRate', Number(e.target.value))}>
          {[0.25, 0.5, 0.75, 1].map((r) => <option key={r} value={r}>{r}×</option>)}
        </select>
      </label>

      {p.hasVideo && (
        <button className="ghost" onClick={p.onNewVideo}>Load another video</button>
      )}
    </section>
  );
}
