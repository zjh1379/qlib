import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useHoldings() {
  return useQuery({
    queryKey: ['portfolio', 'holdings'],
    queryFn: () => api.portfolio.holdings(),
    staleTime: 60_000,
  });
}

export function useTransactions(params: { symbol?: string; from?: string; to?: string } = {}) {
  return useQuery({
    queryKey: ['portfolio', 'transactions', params],
    queryFn: () => api.portfolio.listTransactions(params),
    staleTime: 60_000,
  });
}

export function useAddTransaction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.portfolio.addTransaction,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['portfolio'] }),
  });
}

export function useDeleteTransaction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.portfolio.deleteTransaction(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['portfolio'] }),
  });
}
