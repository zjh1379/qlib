import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { paths } from '@/api/types.gen';

export type RetrainScheduleUpdate =
  paths['/api/scheduling/retrain']['put']['requestBody']['content']['application/json'];

export function useRetrainSchedule() {
  return useQuery({
    queryKey: ['scheduling', 'retrain'],
    queryFn: () => api.scheduling.getRetrain(),
    staleTime: 30_000,
  });
}

export function useUpdateRetrainSchedule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RetrainScheduleUpdate) => api.scheduling.putRetrain(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scheduling', 'retrain'] }),
  });
}

export function useRunRetrainNow() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (force: boolean) => api.scheduling.runNow(force),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scheduling', 'retrain'] }),
  });
}
