import { formatSpeed, trackColor } from '../render/overlay';
import type { TrackSnapshot } from '../tracking/types';

interface Props {
  tracks: TrackSnapshot[];
  selectedId: number | null;
  onSelect(id: number | null): void;
}

export function TrackTable({ tracks, selectedId, onSelect }: Props) {
  const sorted = [...tracks].sort((a, b) => a.id - b.id);
  return (
    <section className="panel tracks">
      <h3>
        Objects in view <span className="muted">({tracks.length})</span>
      </h3>
      {sorted.length === 0 ? (
        <p className="hint">No objects tracked yet.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Type</th>
              <th>Speed</th>
              <th>Heading</th>
              <th>In view</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr
                key={s.id}
                className={`${s.id === selectedId ? 'selected' : ''} ${s.state === 'lost' ? 'lost' : ''}`}
                onClick={() => onSelect(s.id === selectedId ? null : s.id)}
              >
                <td>
                  <span className="swatch" style={{ background: trackColor(s.id) }} />#{s.id}
                </td>
                <td>{s.label}</td>
                <td className="num">{formatSpeed(s)}</td>
                <td>
                  {s.headingDeg === null ? (
                    '–'
                  ) : (
                    <span className="heading">
                      <svg viewBox="0 0 16 16" width="14" height="14" style={{ transform: `rotate(${s.headingDeg}deg)` }}>
                        <path fill="currentColor" d="M8 1l5 13-5-3-5 3z" />
                      </svg>
                      {s.compass} {Math.round(s.headingDeg)}°
                      {s.approach && <span className="muted">· {s.approach}</span>}
                    </span>
                  )}
                </td>
                <td className="num">{Math.max(0, s.t - s.firstSeen).toFixed(1)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint small">
        Speeds are estimates. The scale comes from each object&apos;s typical size (car ≈ 4.5 m, bus ≈ 12
        m, airliner ≈ 38 m) and motion towards/away from the camera from how fast it grows or shrinks,
        assuming a typical camera lens (~75° diagonal). Zoomed footage or unusual vehicles will be off.
      </p>
    </section>
  );
}
