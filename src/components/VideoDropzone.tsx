import { useRef, useState, type DragEvent } from 'react';

interface Props {
  onFile(file: File): void;
}

export function VideoDropzone({ onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith('video/'));
    if (file) onFile(file);
  };

  return (
    <div
      className={`dropzone${over ? ' over' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
    >
      <svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true">
        <path
          fill="currentColor"
          d="M4 5h11a2 2 0 0 1 2 2v2.5l4-2.5v10l-4-2.5V17a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z"
        />
      </svg>
      <h2>Drop a video here</h2>
      <p>or click to choose a file — highway traffic, planes in the sky, boats, birds…</p>
      <p className="muted">Everything runs in your browser. Your video is never uploaded.</p>
      <input
        ref={input}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
    </div>
  );
}
