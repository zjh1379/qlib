import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { api } from '@/api/client';

/** List all recorders with summary. Cached for 60s (fresh between mounts). */
export function useRecorders() {
  return useQuery({
    queryKey: ['evaluation', 'recorders'],
    queryFn: () => api.evaluation.listRecorders(),
    staleTime: 60_000,
  });
}

/** Get the cached eval result for a recorder. 404 = never evaluated. */
export function useEvalResult(recorderId: string | null) {
  return useQuery({
    queryKey: ['evaluation', 'result', recorderId],
    queryFn: () => api.evaluation.getResult(recorderId!),
    enabled: !!recorderId,
    staleTime: Infinity,
    retry: false, // Don't retry on 404
  });
}

/** Trigger evaluation. On success, invalidates the recorders list (so has_eval flips). */
export function useRunEvaluation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { recorder_id: string; top_k?: number; cost_bps?: number; force_refresh?: boolean }) =>
      api.evaluation.run({
        recorder_id: body.recorder_id,
        top_k: body.top_k ?? 30,
        cost_bps: body.cost_bps ?? 10,
        force_refresh: body.force_refresh ?? false,
      }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['evaluation', 'recorders'] });
      qc.setQueryData(['evaluation', 'result', data.recorder_id], data);
    },
  });
}

/** Compare two recorders. Triggers eval on either if not cached. */
export function useCompare(a: string | null, b: string | null) {
  return useQuery({
    queryKey: ['evaluation', 'compare', a, b],
    queryFn: () => api.evaluation.compare(a!, b!),
    enabled: !!a && !!b,
    staleTime: Infinity,
  });
}
