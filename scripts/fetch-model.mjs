// Downloads the YOLOX ONNX models (Apache-2.0, Megvii) into public/models/.
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const RELEASE = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0';
const MODELS = ['yolox_tiny.onnx', 'yolox_s.onnx'];
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dir = resolve(root, 'public/models');
mkdirSync(dir, { recursive: true });

for (const file of MODELS) {
  const target = resolve(dir, file);
  if (existsSync(target) && statSync(target).size > 1_000_000) continue;
  const url = `${RELEASE}/${file}`;
  console.log(`Downloading ${url} ...`);
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    console.error(`Model download failed: HTTP ${res.status}`);
    process.exit(1);
  }
  const tmp = `${target}.part`;
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  renameSync(tmp, target);
  console.log(`Saved ${target} (${(statSync(target).size / 1e6).toFixed(1)} MB)`);
}
