import { useEffect, useState } from 'react';
import { api } from '@/api/client';
import { useJobPolling } from './useJobPolling';

export type ActiveJobKind = 'refresh' | 'retrain' | 'evaluation' | 'inference';

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
 * Polls all backend job-tracking endpoints and returns a unified list.
 *
 * Adaptive polling (added 2026-05-29 after memory audit): when no job is
 * active, slow polling to 30s per endpoint instead of 3-5s. With 4
 * endpoints × 5s that was ~50 req/min just on idle; now ~8 req/min idle.
 * As soon as ANY endpoint reports a running job, all polls flip to 3s
 * for fast progress feedback. Reduces backend RSS pressure from constant
 * candidates / qlib calls.
 */
export function useActiveJobs(): ActiveJob[] {
  const [hasActive, setHasActive] = useState(false);
  const fastMs = 3_000;
  const slowMs = 30_000;
  const interval = hasActive ? fastMs : slowMs;

  // One shared adaptive cadence; each job polls via the same useJobPolling hook.
  const refresh = useJobPolling('refresh', () => api.data.refreshActive(), interval);
  const retrain = useJobPolling('retrain', () => api.data.retrainActive(), interval);
  const evals = useJobPolling('evaluation', () => api.data.evalActive(), interval);
  const inference = useJobPolling('inference', () => api.inference.active(), interval);

  // Watch for any running job and bump polling to fast.
  useEffect(() => {
    const running = (
      (refresh?.status === 'running') ||
      (retrain?.status === 'pending' || retrain?.status === 'running') ||
      (Array.isArray(evals) && evals.length > 0) ||
      (inference?.status === 'running')
    );
    if (running !== hasActive) setHasActive(!!running);
  }, [refresh, retrain, evals, inference, hasActive]);

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

  if (inference) {
    const status = inference.status as ActiveJob['status'];
    const recent =
      inference.started_at && Date.now() - new Date(inference.started_at).getTime() < 60_000;
    if (status === 'running' || (recent && (status === 'done' || status === 'failed'))) {
      out.push({
        kind: 'inference',
        label:
          status === 'running'
            ? '模型推理'
            : status === 'done'
              ? '✓ 推理完成'
              : '✗ 推理失败',
        detail:
          inference.new_rows != null
            ? `+${inference.new_rows} 行`
            : inference.end_date ?? undefined,
        status,
        started_at: inference.started_at,
        href: '/picks',
      });
    }
  }

  return out;
}
