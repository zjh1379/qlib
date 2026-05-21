import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import TransactionForm from '@/portfolio/TransactionForm';

const mutateAsync = vi.fn().mockResolvedValue({
  id: 1,
  symbol: 'SH600519',
  kind: 'buy',
  qty: 100,
  price: 1500,
  fee: 5,
  executed_at: '2026-05-20T10:00:00Z',
  broker: null,
  notes: null,
  created_at: '2026-05-20T10:00:00Z',
});

vi.mock('@/portfolio/hooks', () => ({
  useAddTransaction: () => ({
    mutateAsync,
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

describe('TransactionForm', () => {
  it('renders all fields when open', () => {
    render(wrap(<TransactionForm open={true} onClose={() => {}} />));
    expect(screen.getByLabelText('symbol')).toBeInTheDocument();
    expect(screen.getByLabelText('qty')).toBeInTheDocument();
    expect(screen.getByLabelText('price')).toBeInTheDocument();
    expect(screen.getByLabelText('fee')).toBeInTheDocument();
    expect(screen.getByLabelText('executed_at')).toBeInTheDocument();
    expect(screen.getByLabelText('broker')).toBeInTheDocument();
    expect(screen.getByLabelText('notes')).toBeInTheDocument();
    expect(screen.getByText('买入')).toBeInTheDocument();
    expect(screen.getByText('卖出')).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(wrap(<TransactionForm open={false} onClose={() => {}} />));
    expect(screen.queryByLabelText('symbol')).not.toBeInTheDocument();
  });

  it('shows error for invalid symbol', async () => {
    mutateAsync.mockClear();
    render(wrap(<TransactionForm open={true} onClose={() => {}} />));
    fireEvent.change(screen.getByLabelText('symbol'), { target: { value: 'INVALID' } });
    fireEvent.change(screen.getByLabelText('qty'), { target: { value: '100' } });
    fireEvent.change(screen.getByLabelText('price'), { target: { value: '1500' } });
    fireEvent.click(screen.getByText('提交'));
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/代码格式无效/);
    });
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it('shows error for non-positive quantity', async () => {
    mutateAsync.mockClear();
    render(wrap(<TransactionForm open={true} onClose={() => {}} />));
    fireEvent.change(screen.getByLabelText('symbol'), { target: { value: 'SH600519' } });
    fireEvent.change(screen.getByLabelText('qty'), { target: { value: '0' } });
    fireEvent.change(screen.getByLabelText('price'), { target: { value: '1500' } });
    fireEvent.click(screen.getByText('提交'));
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/数量必须大于 0/);
    });
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it('calls mutation on valid submit', async () => {
    mutateAsync.mockClear();
    const onClose = vi.fn();
    render(wrap(<TransactionForm open={true} onClose={onClose} />));
    fireEvent.change(screen.getByLabelText('symbol'), { target: { value: 'sh600519' } });
    fireEvent.change(screen.getByLabelText('qty'), { target: { value: '100' } });
    fireEvent.change(screen.getByLabelText('price'), { target: { value: '1500' } });
    fireEvent.change(screen.getByLabelText('fee'), { target: { value: '5' } });
    fireEvent.change(screen.getByLabelText('executed_at'), {
      target: { value: '2026-05-20T10:00' },
    });
    fireEvent.click(screen.getByText('提交'));
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1);
    });
    const arg = mutateAsync.mock.calls[0][0];
    expect(arg.symbol).toBe('SH600519');
    expect(arg.qty).toBe(100);
    expect(arg.price).toBe(1500);
    expect(arg.fee).toBe(5);
    expect(arg.kind).toBe('buy');
    expect(onClose).toHaveBeenCalled();
  });
});
