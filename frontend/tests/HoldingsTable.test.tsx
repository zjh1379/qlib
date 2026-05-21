import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import HoldingsTable from '@/portfolio/HoldingsTable';
import type { components } from '@/api/types.gen';

type HoldingsResponse = components['schemas']['HoldingsResponse'];

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient();
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>
  );
}

const emptyData: HoldingsResponse = {
  holdings: [],
  total_cost: 0,
  total_market_value: 0,
  total_unrealized_pnl: 0,
  as_of: null,
};

const sampleData: HoldingsResponse = {
  holdings: [
    {
      symbol: 'SH600519',
      name: '贵州茅台',
      qty: 100,
      avg_cost: 1500,
      effective_cost: 1500,
      current_price: 1650,
      market_value: 165000,
      unrealized_pnl: 15000,
      unrealized_pnl_pct: 0.1,
    },
    {
      symbol: 'SZ300750',
      name: '宁德时代',
      qty: 200,
      avg_cost: 250,
      effective_cost: 250,
      current_price: 240,
      market_value: 48000,
      unrealized_pnl: -2000,
      unrealized_pnl_pct: -0.04,
    },
  ],
  total_cost: 200000,
  total_market_value: 213000,
  total_unrealized_pnl: 13000,
  as_of: '2026-05-20T15:00:00Z',
};

describe('HoldingsTable', () => {
  it('renders empty state when no holdings', () => {
    render(wrap(<HoldingsTable data={emptyData} />));
    expect(screen.getByText(/暂无持仓/)).toBeInTheDocument();
  });

  it('renders rows with symbol, name and qty when holdings present', () => {
    render(wrap(<HoldingsTable data={sampleData} />));
    expect(screen.getByText('SH600519')).toBeInTheDocument();
    expect(screen.getByText('贵州茅台')).toBeInTheDocument();
    expect(screen.getByText('SZ300750')).toBeInTheDocument();
    expect(screen.getByText('宁德时代')).toBeInTheDocument();
  });

  it('renders summary stats', () => {
    render(wrap(<HoldingsTable data={sampleData} />));
    expect(screen.getByText('总投入')).toBeInTheDocument();
    expect(screen.getByText('总市值')).toBeInTheDocument();
    expect(screen.getByText('总浮盈')).toBeInTheDocument();
    expect(screen.getByText('总收益率')).toBeInTheDocument();
  });

  it('symbol cell links to chart page', () => {
    render(wrap(<HoldingsTable data={sampleData} />));
    const link = screen.getByText('SH600519').closest('a');
    expect(link).toHaveAttribute('href', '/charts/SH600519');
  });
});
