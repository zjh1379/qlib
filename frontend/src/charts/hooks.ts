import { useQuery } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';

interface ChartArgs {
  symbol: string;
  start: string;
  end: string;
  withPred?: boolean;
}

export function useChart({ symbol, start, end, withPred = true }: ChartArgs) {
  return useQuery({
    queryKey: ['chart', symbol, start, end, withPred],
    queryFn: () => api.charts.get(symbol, { start, end, with_pred: withPred }),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
    retry: (count, err) => (err instanceof ApiError && err.status === 404 ? false : count < 2),
  });
}
