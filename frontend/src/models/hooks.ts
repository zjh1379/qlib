import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useScreen(
  params: {
    top?: number;
    days?: number;
    min_top?: number;
    view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
    min_price?: number | null;
    max_price?: number | null;
    pct_change_n?: 1 | 3 | 5 | 10 | 20;
    min_pct_change?: number | null;
    max_pct_change?: number | null;
    min_amplitude?: number | null;
    max_amplitude?: number | null;
    min_vol_ratio?: number | null;
    max_vol_ratio?: number | null;
    new_high_n?: 0 | 20 | 60 | 120;
    boards?: string[];
    exclude_st?: boolean;
  } = {},
) {
  return useQuery({
    queryKey: ['models', 'screen', params],
    queryFn: () => api.models.screen(params),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => prev, // keep old results visible during refetch
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

export function useCandidates(
  params: {
    top?: number;
    days?: number;
    min_top?: number;
    view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
  } = {},
) {
  return useQuery({
    queryKey: ['models', 'candidates', params],
    queryFn: () => api.models.candidates(params),
    staleTime: Infinity,         // Candidates rarely change within a session.
    gcTime: 30 * 60_000,         // Keep in memory for 30 min after last use.
    refetchOnWindowFocus: false, // Don't refetch when window regains focus.
    placeholderData: (prev) => prev,
  });
}
