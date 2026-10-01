import type { ModelSpec } from './models';
import type { DetectParams, DetectResponse, WorkerRequest, WorkerResponse } from './protocol';
import type { Backend, DetectorStatus } from './types';

export type DetectResult = DetectResponse;

interface Pending {
  resolve: (r: DetectResult) => void;
  reject: (e: Error) => void;
}

/** Main-thread handle to the detection Web Worker. */
export class DetectorClient {
  private worker!: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private readyPromise: Promise<Backend>;
  private resolveReady!: (b: Backend) => void;
  private rejectReady!: (e: Error) => void;
  status: DetectorStatus = { state: 'idle' };

  constructor(
    readonly model: ModelSpec,
    private modelUrl: string,
    private onStatus: (s: DetectorStatus) => void,
    preferWebGPU = true,
  ) {
    this.readyPromise = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.readyPromise.catch(() => {});
    this.start(preferWebGPU);
  }

  ready(): Promise<Backend> {
    return this.readyPromise;
  }

  detect(bitmap: ImageBitmap, params: DetectParams): Promise<DetectResult> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.post({ type: 'detect', id, bitmap, ...params }, [bitmap]);
    });
  }

  terminate(): void {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('Detector terminated'));
    this.pending.clear();
  }

  private start(preferWebGPU: boolean) {
    this.worker = new Worker(new URL('./detector.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.onMessage(e.data);
    this.worker.onerror = (e) => {
      const message = e.message || 'Detector worker crashed';
      this.setStatus({ state: 'error', message });
      this.rejectReady(new Error(message));
    };
    this.setStatus({ state: 'loading', message: 'Loading model…' });
    this.post({ type: 'init', modelUrl: this.modelUrl, inputSize: this.model.inputSize, preferWebGPU });
  }

  private onMessage(msg: WorkerResponse) {
    switch (msg.type) {
      case 'progress':
        this.setStatus({ state: 'loading', message: msg.message });
        break;
      case 'ready':
        this.setStatus({ state: 'ready', backend: msg.backend });
        this.resolveReady(msg.backend);
        break;
      case 'fallback':
        this.worker.terminate();
        this.start(false);
        break;
      case 'result': {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        p?.resolve({ detections: msg.detections, inferMs: msg.inferMs, camera: msg.camera });
        break;
      }
      case 'error':
        if (msg.id !== undefined) {
          const p = this.pending.get(msg.id);
          this.pending.delete(msg.id);
          p?.reject(new Error(msg.message));
        } else {
          this.setStatus({ state: 'error', message: msg.message });
          this.rejectReady(new Error(msg.message));
        }
        break;
    }
  }

  private post(msg: WorkerRequest, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  private setStatus(s: DetectorStatus) {
    this.status = s;
    this.onStatus(s);
  }
}
