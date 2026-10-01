/**
 * Minimum-cost assignment (Hungarian / Kuhn–Munkres, O(n³)) for a rectangular cost
 * matrix. Pairs whose cost is ≥ `maxCost` are treated as forbidden and never returned.
 * Returns the matched [row, col] pairs.
 */
export function assign(cost: number[][], maxCost: number): Array<[number, number]> {
  const rows = cost.length;
  const cols = rows ? cost[0].length : 0;
  if (!rows || !cols) return [];
  const n = Math.max(rows, cols);
  const BIG = maxCost * 10 + 1e6;
  const at = (i: number, j: number) => {
    if (i >= rows || j >= cols) return BIG;
    const c = cost[i][j];
    return c >= maxCost || !Number.isFinite(c) ? BIG : c;
  };

  // Classic potentials-based implementation (1-indexed).
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1); // p[j] = row matched to column j
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = at(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const pairs: Array<[number, number]> = [];
  for (let j = 1; j <= n; j++) {
    const i = p[j] - 1;
    const jj = j - 1;
    if (i < rows && jj < cols && at(i, jj) < BIG) pairs.push([i, jj]);
  }
  return pairs;
}
