import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

/** Manually trigger AI analysis for the current top-N picks.
 * Idempotent server-side: picks already analyzed ok for the date are skipped,
 * so this cheaply fills gaps (new date / prior partial / failed). */
export function useTriggerAnalysis() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.analysis.runNow(),
    onSuccess: async (resp) => {
      const { toast } = await import('@/jobs/toast');
      if (resp.status === 'disabled') {
        toast.info('AI 解读未启用（需配置 API key 并打开开关）');
      } else if (resp.status === 'already_running') {
        toast.info('已有 AI 解读任务进行中…');
      } else {
        toast.info('已触发 AI 解读，等待完成…');
      }
      qc.invalidateQueries({ queryKey: ['analysis', 'active'] });
    },
  });
}
