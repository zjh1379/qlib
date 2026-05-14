import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useDataStatus() {
  return useQuery({
    queryKey: ['data', 'status'],
    queryFn: () => api.data.status(),
    staleTime: 30_000,
    refetchInterval: 60_000, // poll every minute to detect freshness drift
  });
}

export function useInstruments(market = 'csi300') {
  return useQuery({
    queryKey: ['instruments', market],
    queryFn: () => api.instruments(market),
    staleTime: 60 * 60_000, // 1 hour — instruments rarely change
  });
}

export function useRefreshData() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.data.refresh(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['data'] });
    },
  });
}

export function useRefreshJob(jobId: string | null) {
  return useQuery({
    queryKey: ['data', 'refresh', jobId],
    queryFn: () => api.data.refreshStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'running' ? 3_000 : false, // poll every 3s while running
  });
}

export function useMarkets() {
  return useQuery({
    queryKey: ['data', 'markets'],
    queryFn: () => api.data.markets(),
    staleTime: 60 * 60_000,
  });
}

export function useAddSymbol() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (symbol: string) => api.data.addSymbol(symbol),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['instruments'] });
      qc.invalidateQueries({ queryKey: ['data', 'markets'] });
      qc.invalidateQueries({ queryKey: ['data', 'status'] });
    },
  });
}
