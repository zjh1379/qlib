import { describe, it, expect } from 'vitest';
import { selectCandidates } from '@/pages/picks/filter';
import { DEFAULT_FILTERS } from '@/pages/picks/types';
import type { components } from '@/api/types.gen';

type Candidate = components['schemas']['ScreenItem'];

function mk(symbol: string, daily_ranks: (number | null)[], daily_scores: (number | null)[]): Candidate {
  return {
    rank: 0, symbol, name: symbol, score_today: 0, score_avg: 0, rank_avg: 0,
    days_in_top: 0, consensus: 0, base_scores: {}, horizons: {},
    last_price: 10, daily_ranks, daily_scores, is_st: false, board: 'main',
  } as Candidate;
}

const base = {
  ...DEFAULT_FILTERS, top: 2, days: 3, min_top: 2,
  min_price: null, max_price: null, exclude_st: false, min_consensus: 0,
};

describe('selectCandidates', () => {
  it('keeps only symbols in top-N on >= min_top of last D days', () => {
    const cands = [
      mk('A', [1, 1, 1], [0.5, 0.5, 0.6]),
      mk('B', [5, 1, 2], [0.1, 0.2, 0.3]),
      mk('C', [9, 9, 9], [0.9, 0.9, 0.9]),
    ];
    const syms = selectCandidates(cands, base, 3).map((c) => c.symbol);
    expect(syms).toContain('A');
    expect(syms).toContain('B');
    expect(syms).not.toContain('C');
  });

  it('orders by window score_avg desc and assigns 1-based rank', () => {
    const cands = [
      mk('LOW', [1, 1, 1], [0.1, 0.1, 0.1]),
      mk('HIGH', [1, 1, 1], [0.9, 0.9, 0.9]),
    ];
    const out = selectCandidates(cands, { ...base, min_top: 1 }, 3);
    expect(out[0].symbol).toBe('HIGH');
    expect(out[0].rank).toBe(1);
    expect(out[1].rank).toBe(2);
  });

  it('caps displayed rows to `top`', () => {
    const cands = [
      mk('A', [1, 1, 1], [0.5, 0.5, 0.5]),
      mk('B', [1, 1, 1], [0.4, 0.4, 0.4]),
      mk('C', [1, 1, 1], [0.3, 0.3, 0.3]),
    ];
    const out = selectCandidates(cands, { ...base, top: 2, min_top: 1 }, 3);
    expect(out.length).toBe(2);
    expect(out.map((c) => c.symbol)).toEqual(['A', 'B']);
  });
});
