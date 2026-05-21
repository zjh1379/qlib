import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useScreen(params: { top?: number; days?: number; min_top?: number } = {}) {
  return useQuery({
    queryKey: ['models', 'screen', params],
    queryFn: () => api.models.screen(params),
    staleTime: 5 * 60_000,
  });
}

export function usePredictionHistory(symbol: string, days = 60) {
  return useQuery({
    queryKey: ['models', 'predictions', symbol, days],
    queryFn: () => api.models.predictions(symbol, { days }),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
  });
}
