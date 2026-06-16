import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { api } from '@/api/client';

export function useStartTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (force: boolean) => api.training.run(force),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training', 'active'] }),
  });
}

export function useActiveTrainingJob() {
  return useQuery({
    queryKey: ['training', 'active'],
    queryFn: () => api.training.active(),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === 'running' || s === 'pending' ? 3_000 : false;
    },
    refetchIntervalInBackground: true,
  });
}

export function useTrainingJobDetail(jobId: string | null) {
  return useQuery({
    queryKey: ['training', 'job', jobId],
    queryFn: () => api.training.status(jobId as string),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 3_000 : false),
    refetchIntervalInBackground: true,
  });
}

export function useTrainingRuns() {
  return useQuery({
    queryKey: ['training', 'runs'],
    queryFn: () => api.training.runs(),
    staleTime: 10_000,
  });
}

export function useRollback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (target: 'previous_1' | 'previous_2') => api.models.rollback(target),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['training', 'runs'] });
      qc.invalidateQueries({ queryKey: ['evaluation', 'recorders'] });
    },
  });
}
