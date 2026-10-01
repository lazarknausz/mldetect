import { useEffect, useState, type RefObject, type MouseEvent } from 'react';

interface Props {
  src: string;
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  disabled: boolean;
  onLoaded(): void;
  onStageClick(x: number, y: number): void;
}

function fmt(t: number) {
  if (!Number.isFinite(t)) return '0:00';
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function VideoStage({ src, videoRef, canvasRef, disabled, onLoaded, onStageClick }: Props) {
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const sync = () => {
      setPlaying(!v.paused && !v.ended);
      setTime(v.currentTime);
      setDuration(v.duration || 0);
    };
    const events = ['play', 'pause', 'ended', 'timeupdate', 'seeked', 'loadedmetadata', 'durationchange'];
    events.forEach((ev) => v.addEventListener(ev, sync));
    return () => events.forEach((ev) => v.removeEventListener(ev, sync));
  }, [videoRef, src]);

  const toggle = () => {
    const v = videoRef.current;
    if (!v || disabled) return;
    if (v.paused || v.ended) void v.play();
    else v.pause();
  };

  const click = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onStageClick(e.clientX - r.left, e.clientY - r.top);
  };

  return (
    <div className="stage-wrap">
      <div className="stage" onClick={click} onDoubleClick={toggle}>
        <video
          ref={videoRef}
          src={src}
          muted
          playsInline
          preload="auto"
          onLoadedData={onLoaded}
        />
        <canvas ref={canvasRef} className="overlay" />
      </div>
      <div className="transport">
        <button className="icon-btn" onClick={toggle} disabled={disabled} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? (
            <svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M8 5v14l11-7z" /></svg>
          )}
        </button>
        <span className="time">{fmt(time)}</span>
        <input
          className="seek"
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={Math.min(time, duration || 0)}
          disabled={disabled}
          onChange={(e) => {
            const v = videoRef.current;
            if (v) v.currentTime = Number(e.target.value);
          }}
          aria-label="Seek"
        />
        <span className="time">{fmt(duration)}</span>
      </div>
    </div>
  );
}
