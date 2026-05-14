import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import SymbolSearch from '@/components/SymbolSearch';

vi.mock('@/data/hooks', () => ({
  useInstruments: () => ({
    data: {
      market: 'all',
      count: 3,
      items: [
        { symbol: 'SH600519', name: '贵州茅台' },
        { symbol: 'SH600036', name: '招商银行' },
        { symbol: 'SZ300750', name: '宁德时代' },
      ],
    },
    isPending: false,
  }),
  useAddSymbol: () => ({
    mutateAsync: vi
      .fn()
      .mockResolvedValue({ symbol: 'SH601398', fetched_rows: 100, message: 'ok' }),
    isPending: false,
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

describe('SymbolSearch', () => {
  it('renders input with aria-label', () => {
    render(wrap(<SymbolSearch />));
    expect(screen.getByLabelText('symbol search')).toBeInTheDocument();
  });

  it('filters by Chinese name', async () => {
    render(wrap(<SymbolSearch />));
    const input = screen.getByLabelText('symbol search');
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '茅台' } });
    await waitFor(() => {
      expect(screen.getByText(/SH600519/)).toBeInTheDocument();
    });
  });

  it('filters by symbol code', async () => {
    render(wrap(<SymbolSearch />));
    const input = screen.getByLabelText('symbol search');
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: '600036' } });
    await waitFor(() => {
      expect(screen.getByText(/招商银行/)).toBeInTheDocument();
    });
  });

  it('shows add-to-download prompt for valid symbol not in dataset', async () => {
    render(wrap(<SymbolSearch />));
    const input = screen.getByLabelText('symbol search');
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: 'SH601398' } });
    await waitFor(() => {
      expect(screen.getByText(/到下载列表/)).toBeInTheDocument();
      expect(screen.getByText(/点击下载/)).toBeInTheDocument();
      // The valid-symbol prompt highlights the upper-cased code in a <span>
      expect(screen.getAllByText(/SH601398/).length).toBeGreaterThan(0);
    });
  });
});
