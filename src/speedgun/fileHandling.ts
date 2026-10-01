/**
 * File handling: load a local video into the (hidden) <video> element, find its frame
 * rate, and seek to exact frames. Nothing is uploaded — the browser reads the file
 * from disk through an object URL.
 */

import { detectFps, frameIndexAt, seekTimeForFrame } from './speedMath';

export interface VideoInfo {
  name: string;
  width: number;
  height: number;
  duration: number;
}

let currentUrl: string | null = null;

/** Loads `file` into `video` and resolves once the first frame can be drawn. */
export function loadVideoFile(file: File, video: HTMLVideoElement): Promise<VideoInfo> {
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = URL.createObjectURL(file);
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener('loadeddata', onLoaded);
      video.removeEventListener('error', onError);
    };
    const onLoaded = () => {
      cleanup();
      resolve({ name: file.name, width: video.videoWidth, height: video.videoHeight, duration: video.duration });
    };
    const onError = () => {
      cleanup();
      reject(new Error(`This browser cannot play "${file.name}". Try an .mp4 (H.264) or .webm file.`));
    };
    video.addEventListener('loadeddata', onLoaded);
    video.addEventListener('error', onError);
    video.muted = true;
    video.src = currentUrl!;
    video.load();
  });
}

/**
 * The browser does not expose a video's FPS, so measure it: play ~1 s muted and record
 * the presentation time of every frame (`requestVideoFrameCallback`), then take the
 * typical gap between frames. Falls back to 30 fps if that is not possible.
 */
export async function detectVideoFps(video: HTMLVideoElement): Promise<number> {
  if (!('requestVideoFrameCallback' in video)) return 30;
  const times: number[] = [];
  const start = video.currentTime;
  await new Promise<void>((resolve) => {
    let handle = 0;
    const timer = setTimeout(done, 1500);
    function done() {
      clearTimeout(timer);
      video.cancelVideoFrameCallback(handle);
      video.pause();
      resolve();
    }
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      times.push(meta.mediaTime);
      if (times.length >= 25) done();
      else handle = video.requestVideoFrameCallback(onFrame);
    };
    handle = video.requestVideoFrameCallback(onFrame);
    video.playbackRate = 1;
    video.play().catch(done);
  });
  video.currentTime = start;
  return detectFps(times) ?? 30;
}

/**
 * Seeks so that frame `n` is on screen and resolves once it is decoded. We seek to the
 * *middle* of the frame's time slot (n + 0.5) / FPS so rounding can never land on the
 * neighbouring frame. Resolves with the index of the frame actually shown.
 */
export function seekToFrame(video: HTMLVideoElement, n: number, fps: number): Promise<number> {
  return new Promise((resolve) => {
    let done = false;
    let handle = 0;
    const target = Math.min(seekTimeForFrame(n, fps), Math.max(0, video.duration - 0.5 / fps));
    const finish = (mediaTime: number) => {
      if (done) return;
      done = true;
      video.cancelVideoFrameCallback(handle);
      video.removeEventListener('seeked', onSeeked);
      resolve(frameIndexAt(mediaTime, fps));
    };
    // `seeked` can fire before the new frame is composited; give the frame callback a
    // moment and fall back to the requested time.
    const onSeeked = () => setTimeout(() => finish(video.currentTime), 300);
    handle = video.requestVideoFrameCallback((_now, meta) => finish(meta.mediaTime));
    video.addEventListener('seeked', onSeeked);
    video.currentTime = target;
  });
}

export function frameCount(video: HTMLVideoElement, fps: number): number {
  return Math.max(1, Math.floor(video.duration * fps));
}
