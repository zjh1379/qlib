import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Picks from '@/pages/Picks';

vi.mock('@/models/hooks', () => ({
  useScreen: () => ({
    data: {
      experiment: 'daily_cn_fresh',
      recorder_id: 'abc123',
      latest_date: '2026-05-08',
      window_days: 5,
      universe_size: 300,
      items: [
        {
          rank: 1,
          symbol: 'SH600519',
          name: '贵州茅台',
          score_today: 0.05,
          score_avg: 0.04,
          rank_avg: 12.5,
          days_in_top: 5,
        },
        {
          rank: 2,
          symbol: 'SH600036',
          name: '招商银行',
          score_today: 0.03,
          score_avg: 0.03,
          rank_avg: 25.0,
          days_in_top: 4,
        },
      ],
    },
    isPending: false,
    error: null,
  }),
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient();
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Picks', () => {
  it('renders header and filter inputs', () => {
    render(wrap(<Picks />));
    expect(screen.getByText(/选股工作台/)).toBeInTheDocument();
    expect(screen.getByText(/^Top N$/)).toBeInTheDocument();
    expect(screen.getByText(/窗口天数/)).toBeInTheDocument();
    expect(screen.getByText(/最少进 top N 天数/)).toBeInTheDocument();
  });

  it('renders result rows with symbols and names', () => {
    render(wrap(<Picks />));
    expect(screen.getByText('SH600519')).toBeInTheDocument();
    expect(screen.getByText('贵州茅台')).toBeInTheDocument();
    expect(screen.getByText('SH600036')).toBeInTheDocument();
    expect(screen.getByText('招商银行')).toBeInTheDocument();
  });

  it('renders footer info from screen response', () => {
    render(wrap(<Picks />));
    expect(screen.getByText('daily_cn_fresh')).toBeInTheDocument();
    expect(screen.getByText('2026-05-08')).toBeInTheDocument();
  });
});
