import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Picks from '@/pages/Picks';

const baseItem = {
  rank: 1, symbol: 'SH600519', name: '贵州茅台', score_today: 0.05, score_avg: 0.04,
  rank_avg: 1, days_in_top: 5, consensus: 1, base_scores: {}, horizons: {},
  last_price: 10, daily_ranks: [1, 1, 1, 1, 1], daily_scores: [0.04, 0.04, 0.04, 0.04, 0.04],
  is_st: false, board: 'main',
};
const item2 = {
  ...baseItem, rank: 2, symbol: 'SH600036', name: '招商银行', score_avg: 0.03, last_price: 20,
  daily_ranks: [2, 2, 2, 2, 2], daily_scores: [0.03, 0.03, 0.03, 0.03, 0.03],
};

vi.mock('@/models/hooks', () => ({
  useCandidates: () => ({
    data: {
      experiment: 'rolling_v2_ensemble', recorder_id: 'abc123', latest_date: '2026-06-16',
      window_days: 20, universe_size: 800, items: [baseItem, item2],
      available_models: ['lgbm_1d', 'lgbm_5d', 'alstm_5d'], active_models: null,
      window_dates: ['2026-06-10', '2026-06-11', '2026-06-12', '2026-06-13', '2026-06-16'],
      as_of_date: '2026-06-16', data_latest_date: '2026-06-16', data_stale_days: 0,
    },
    isPending: false, isFetching: false, error: null,
  }),
}));

// Mock useRecompute so the applied combo is "warmed" -> the GET-gated rows render,
// and no real network call fires on mount.
vi.mock('@/pages/picks/useRecompute', () => ({
  useRecompute: () => ({
    isWarmed: () => true,
    start: async () => {},
    job: null,
    elapsedSec: 0,
  }),
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient();
  return <QueryClientProvider client={qc}><MemoryRouter>{ui}</MemoryRouter></QueryClientProvider>;
}

describe('Picks', () => {
  it('renders header and the recompute section', () => {
    render(wrap(<Picks />));
    expect(screen.getByText(/选股工作台/)).toBeInTheDocument();
    expect(screen.getByText(/需重新计算/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /重新计算/ })).toBeInTheDocument();
  });

  it('renders both candidate rows by default (loose filters)', () => {
    render(wrap(<Picks />));
    expect(screen.getByText('SH600519')).toBeInTheDocument();
    expect(screen.getByText('SH600036')).toBeInTheDocument();
  });

  it('recompute button starts disabled (no draft change yet)', () => {
    render(wrap(<Picks />));
    expect(screen.getByRole('button', { name: /重新计算/ })).toBeDisabled();
  });
});
