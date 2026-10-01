import { describe, expect, it } from 'vitest';
import { assign } from './assignment';

const sorted = (p: Array<[number, number]>) => [...p].sort((a, b) => a[0] - b[0]);

describe('assign', () => {
  it('finds the optimal assignment, not the greedy one', () => {
    // Greedy would take (0,0)=1 then (1,1)=10 → 11; optimum is (0,1)+(1,0) = 2+3 = 5.
    expect(sorted(assign([[1, 2], [3, 10]], 100))).toEqual([[0, 1], [1, 0]]);
  });
  it('handles rectangular matrices', () => {
    expect(sorted(assign([[5, 1, 9]], 100))).toEqual([[0, 1]]);
    expect(sorted(assign([[5], [1], [9]], 100))).toEqual([[1, 0]]);
  });
  it('never returns forbidden pairs', () => {
    expect(assign([[Infinity, 0.5], [0.2, Infinity]], 1)).toHaveLength(2);
    expect(assign([[2, 3]], 1)).toEqual([]);
    expect(assign([], 1)).toEqual([]);
  });
});
