import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';

export type ActiveJobKind = 'refresh' | 'retrain' | 'evaluation';

export interface ActiveJob {
  kind: ActiveJobKind;
  label: string;          // human-readable, e.g. '刷新中', '重训中', '评估中'
  detail?: string;        // optional sub-info, e.g. '23/180' or recorder_id[:8]
  status: 'running' | 'done' | 'failed';
  started_at?: string | null;
  /** Link to navigate to when user clicks the badge — varies per kind. */
  href: string;
}

/**
 * Polls all backend job-tracking endpoints (data refresh, retrain, evaluation)
 * and returns a unified list of active or recently-finished jobs.
 *
 * Each endpoint has its own /active/peek route returning the most recent
 * non-terminal (or just-terminal) job; this hook fans out and merges them.
 *
 * Used by <ActiveJobsBadge /> in Layout so the user can see progress
 * regardless of which page they're on. 5s polling — enough granularity
 * for ~30s eval calls / ~3 min refresh / 1.5h retrain.
 */
export function useActiveJobs(): ActiveJob[] {
  const { data: refresh } = useQuery({
    queryKey: ['jobs', 'refresh', 'active'],
    queryFn: () => api.data.refreshActive(),
    staleTime: 5_000,
    refetchInterval: 5_000,
    refetchIntervalInBackground: true,
  });
  const { data: retrain } = useQuery({
    queryKey: ['jobs', 'retrain', 'active'],
    queryFn: () => api.data.retrainActive(),
    staleTime: 5_000,
    refetchInterval: 5_000,
    refetchIntervalInBackground: true,
  });
  const { data: evals } = useQuery({
    queryKey: ['jobs', 'evaluation', 'active'],
    queryFn: () => api.data.evalActive(),
    staleTime: 5_000,
    refetchInterval: 5_000,
    refetchIntervalInBackground: true,
  });

  const out: ActiveJob[] = [];

  if (refresh) {
    const status = refresh.status as ActiveJob['status'];
    // Only show refresh chip if running OR finished within ~30s.
    const recent =
      refresh.started_at && Date.now() - new Date(refresh.started_at).getTime() < 60_000;
    if (status === 'running' || (recent && (status === 'done' || status === 'failed'))) {
      out.push({
        kind: 'refresh',
        label: status === 'running' ? '数据刷新' : status === 'done' ? '✓ 数据已刷新' : '✗ 刷新失败',
        status: status === 'running' ? 'running' : status === 'done' ? 'done' : 'failed',
        started_at: refresh.started_at,
        href: '/',
      });
    }
  }

  if (retrain) {
    const status =
      retrain.status === 'pending' || retrain.status === 'running'
        ? ('running' as const)
        : retrain.status === 'done'
          ? ('done' as const)
          : retrain.status === 'failed'
            ? ('failed' as const)
            : null;
    const recent =
      retrain.started_at && Date.now() - new Date(retrain.started_at).getTime() < 60_000;
    if (status === 'running' || (status && recent)) {
      out.push({
        kind: 'retrain',
        label:
          status === 'running'
            ? `重训 (${retrain.kind === 'manual' ? '手动' : '定时'})`
            : status === 'done'
              ? '✓ 重训完成'
              : '✗ 重训失败',
        detail: retrain.error ?? undefined,
        status,
        started_at: retrain.started_at,
        href: '/settings',
      });
    }
  }

  if (evals && evals.length > 0) {
    out.push({
      kind: 'evaluation',
      label: `评估中 (${evals.length})`,
      detail: evals.map(e => e.recorder_id.slice(0, 8)).join(', '),
      status: 'running',
      started_at: evals[0]?.started_at,
      href: '/evaluation',
    });
  }

  return out;
}
