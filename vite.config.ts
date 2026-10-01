/// <reference types="vitest/config" />
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

const require = createRequire(import.meta.url);
const root = dirname(fileURLToPath(import.meta.url));
const ortDist = dirname(require.resolve('onnxruntime-web'));
const ORT_RUNTIME_FILES = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];

/**
 * Serves ONNX Runtime's WASM loader as plain static files (public/ort/). Using the
 * non-bundled ORT build lets its multi-threaded WASM spawn pthread workers from the real
 * loader script instead of from our bundled detector worker.
 */
function onnxRuntimeFiles(): Plugin {
  return {
    name: 'onnxruntime-files',
    buildStart() {
      const out = resolve(root, 'public/ort');
      mkdirSync(out, { recursive: true });
      for (const f of ORT_RUNTIME_FILES) {
        const src = resolve(ortDist, f);
        const dst = resolve(out, f);
        if (!existsSync(dst) || statSync(dst).size !== statSync(src).size) copyFileSync(src, dst);
      }
    },
  };
}

// Cross-origin isolation lets ONNX Runtime Web use multi-threaded WASM.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // Set BASE_PATH=/mldetect/ when deploying to GitHub Pages.
  base: process.env.BASE_PATH ?? '/',
  plugins: [react(), onnxRuntimeFiles()],
  build: {
    rollupOptions: {
      // Two pages: the object tracker and the calibrated speed gun.
      input: { main: resolve(root, 'index.html'), speedGun: resolve(root, 'speed-gun.html') },
    },
  },
  resolve: {
    alias: { 'onnxruntime-web/webgpu': resolve(ortDist, 'ort.webgpu.min.mjs') },
  },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
