import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

/** Poll the currently running inference job (if any). */
export function useActiveInferenceJob() {
  return useQuery({
    queryKey: ['inference', 'active'],
    queryFn: () => api.inference.active(),
    staleTime: 3_000,
    refetchInterval: 3_000,
    refetchIntervalInBackground: true,
  });
}

/** Last-run / last-success / last-error meta. Refreshed less often. */
export function useInferenceStatus() {
  return useQuery({
    queryKey: ['inference', 'status'],
    queryFn: () => api.inference.status(),
    staleTime: 10_000,
  });
}

/** Manually trigger inference. Shows a sticky info toast on success. */
export function useTriggerInference() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (force?: boolean) => api.inference.runNow(force ?? false),
    onSuccess: async (resp) => {
      const { toast } = await import('@/jobs/toast');
      if (resp.status === 'already_running') {
        toast.info('已有推理任务进行中…');
      } else {
        toast.info('已触发推理任务，等待完成…');
      }
      qc.invalidateQueries({ queryKey: ['inference', 'active'] });
    },
  });
}
