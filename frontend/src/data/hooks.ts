import { useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

const ACTIVE_JOB_LS_KEY = 'qlib.activeRefreshJobId';

export function useDataStatus() {
  return useQuery({
    queryKey: ['data', 'status'],
    queryFn: () => api.data.status(),
    staleTime: 30_000,
    refetchInterval: 60_000, // poll every minute to detect freshness drift
  });
}

export function useInstruments(market = 'csi300') {
  return useQuery({
    queryKey: ['instruments', market],
    queryFn: () => api.instruments(market),
    staleTime: 60 * 60_000, // 1 hour — instruments rarely change
  });
}

export function useRefreshData() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.data.refresh(),
    onSuccess: (r) => {
      // Persist the new job_id so navigating away + back recovers progress.
      try {
        if (r && (r as { job_id?: string }).job_id) {
          localStorage.setItem(ACTIVE_JOB_LS_KEY, (r as { job_id: string }).job_id);
        }
      } catch {
        /* ignore — SSR / private mode */
      }
      qc.invalidateQueries({ queryKey: ['data'] });
    },
  });
}

export function useRefreshJob(jobId: string | null) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ['data', 'refresh', jobId],
    queryFn: () => api.data.refreshStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'running' ? 3_000 : false, // poll every 3s while running
    refetchIntervalInBackground: true,   // continue polling when tab loses focus
  });
  // Cleanup localStorage once the job terminates so the next mount doesn't
  // resurrect a stale done/failed indicator forever.
  useEffect(() => {
    const status = query.data?.status;
    if (status === 'done' || status === 'failed') {
      try {
        const stored = localStorage.getItem(ACTIVE_JOB_LS_KEY);
        if (stored && stored === jobId) {
          // Keep it visible for ~30s after completion so user sees the
          // ✓/✗ on any page they navigate to, then clean up.
          setTimeout(() => {
            try {
              if (localStorage.getItem(ACTIVE_JOB_LS_KEY) === jobId) {
                localStorage.removeItem(ACTIVE_JOB_LS_KEY);
                qc.invalidateQueries({ queryKey: ['data', 'refresh', 'active'] });
              }
            } catch { /* ignore */ }
          }, 30_000);
        }
      } catch { /* ignore */ }
    }
  }, [query.data?.status, jobId, qc]);
  return query;
}

/**
 * Source-of-truth for "is there a refresh job we should care about?".
 *
 * Resolution order:
 *   1. localStorage `qlib.activeRefreshJobId` (set by useRefreshData on
 *      mutation success) — fastest, no roundtrip.
 *   2. Backend `/api/data/refresh/active/peek` — authoritative; recovers
 *      after a browser refresh / localStorage clear.
 *
 * Returns a job_id (string) or null. The caller wires this into
 * `useRefreshJob` to get the actual progress detail.
 */
export function useActiveRefreshJob(): string | null {
  // Read localStorage synchronously to avoid one render with no jobId.
  let lsJobId: string | null = null;
  try {
    lsJobId = localStorage.getItem(ACTIVE_JOB_LS_KEY);
  } catch { /* ignore */ }
  // Always query backend so we recover after browser refresh.
  const { data: serverActive } = useQuery({
    queryKey: ['data', 'refresh', 'active'],
    queryFn: () => api.data.refreshActive(),
    staleTime: 5_000,
    refetchInterval: 5_000,
    refetchIntervalInBackground: true,
  });
  const serverJobId = serverActive?.status === 'running'
    ? serverActive.job_id
    : null;
  // Prefer the server's running job (authoritative); fall back to
  // localStorage so a just-started job is visible before the next poll.
  return serverJobId ?? lsJobId;
}

export function useMarkets() {
  return useQuery({
    queryKey: ['data', 'markets'],
    queryFn: () => api.data.markets(),
    staleTime: 60 * 60_000,
  });
}

export function useAddSymbol() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (symbol: string) => {
      // Issue an immediate "添加中" toast that survives page navigation
      // — add-symbol blocks for 10-30s on baostock + dump_bin, and a user
      // who switches pages mid-wait otherwise has no feedback.
      const { toast, dismiss } = await import('@/jobs/toast');
      const toastId = toast.info(`正在添加 ${symbol} (下载历史 K 线 …~30s)`, -1);
      try {
        const result = await api.data.addSymbol(symbol);
        dismiss(toastId);
        toast.success(`${symbol} 已添加 (${result.fetched_rows} 行)`);
        return result;
      } catch (e) {
        dismiss(toastId);
        throw e;  // re-throw so MutationCache.onError shows the error toast
      }
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['instruments'] });
      qc.invalidateQueries({ queryKey: ['data', 'markets'] });
      qc.invalidateQueries({ queryKey: ['data', 'status'] });
    },
    // Custom toasts above; let the global MutationCache.onError handle
    // additional surfacing of the same error (extra red banner is fine).
  });
}
