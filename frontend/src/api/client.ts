import type { paths } from '@/api/types.gen';

const BASE = ''; // empty in production (same origin); Vite proxy handles /api in dev

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public detail: string,
    public context: Record<string, unknown> = {},
  ) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let body: { detail?: string; code?: string; context?: Record<string, unknown> } = {};
    try {
      body = await res.json();
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, body.code ?? 'unknown', body.detail ?? res.statusText, body.context ?? {});
  }
  return res.json() as Promise<T>;
}

export const api = {
  charts: {
    get: (
      symbol: string,
      params: { start: string; end: string; with_pred?: boolean; experiment?: string },
    ) => {
      const q = new URLSearchParams({
        start: params.start,
        end: params.end,
        with_pred: String(params.with_pred ?? true),
        ...(params.experiment ? { experiment: params.experiment } : {}),
      });
      type R = paths['/api/charts/{symbol}']['get']['responses']['200']['content']['application/json'];
      return request<R>(`/api/charts/${encodeURIComponent(symbol)}?${q.toString()}`);
    },
  },
  ops: {
    health: () => {
      type R = paths['/api/ops/health']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/ops/health');
    },
    memory: () =>
      request<{
        system: {
          ram_total_gb: number; ram_used_gb: number; ram_available_gb: number;
          pagefile_total_gb: number; pagefile_used_gb: number;
          commit_used_gb: number; commit_total_gb: number; commit_pct: number;
          warning: boolean; critical: boolean;
        };
        project_processes: Array<{
          pid: number; name: string; rss_mb: number; vms_mb: number;
          is_self: boolean; cmdline?: string;
        }>;
        other_processes_top10: Array<{
          pid: number; name: string; rss_mb: number; vms_mb: number;
        }>;
        in_memory_state: Record<string, number>;
      }>('/api/ops/memory'),
  },
  data: {
    status: () => {
      type R = paths['/api/data/status']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/data/status');
    },
    refresh: () => {
      type R = paths['/api/data/refresh']['post']['responses']['200']['content']['application/json'];
      return request<R>('/api/data/refresh', { method: 'POST' });
    },
    refreshStatus: (jobId: string) => {
      type R =
        paths['/api/data/refresh/{job_id}']['get']['responses']['200']['content']['application/json'];
      return request<R>(`/api/data/refresh/${encodeURIComponent(jobId)}`);
    },
    refreshActive: () => {
      // Returns {job_id, status, started_at} or null. Used by
      // useActiveRefreshJob() on mount to recover progress after a page
      // navigation.
      return request<{ job_id: string; status: string; started_at: string } | null>(
        '/api/data/refresh/active/peek',
      );
    },
    retrainActive: () =>
      request<{
        job_id: string;
        kind: 'cron' | 'manual';
        status: 'pending' | 'running' | 'done' | 'failed' | 'skipped';
        queued_at: string | null;
        started_at: string | null;
        finished_at: string | null;
        error: string | null;
      } | null>('/api/scheduling/retrain/jobs/active/peek'),
    evalActive: () =>
      request<Array<{ recorder_id: string; started_at: string }>>(
        '/api/evaluation/active/peek',
      ),
    markets: () => {
      type R = paths['/api/data/markets']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/data/markets');
    },
    addSymbol: (symbol: string) => {
      type R = paths['/api/data/symbols/add']['post']['responses']['200']['content']['application/json'];
      return request<R>('/api/data/symbols/add', {
        method: 'POST',
        body: JSON.stringify({ symbol }),
      });
    },
  },
  instruments: (market = 'csi300') => {
    type R = paths['/api/instruments']['get']['responses']['200']['content']['application/json'];
    return request<R>(`/api/instruments?market=${encodeURIComponent(market)}`);
  },
  portfolio: {
    holdings: () => {
      type R =
        paths['/api/portfolio/holdings']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/portfolio/holdings');
    },
    listTransactions: (params: { symbol?: string; from?: string; to?: string } = {}) => {
      type R =
        paths['/api/portfolio/transactions']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      if (params.symbol) q.set('symbol', params.symbol);
      if (params.from) q.set('from', params.from);
      if (params.to) q.set('to', params.to);
      const qs = q.toString();
      return request<R>(`/api/portfolio/transactions${qs ? '?' + qs : ''}`);
    },
    addTransaction: (
      body: paths['/api/portfolio/transactions']['post']['requestBody']['content']['application/json'],
    ) => {
      type R =
        paths['/api/portfolio/transactions']['post']['responses']['200']['content']['application/json'];
      return request<R>('/api/portfolio/transactions', {
        method: 'POST',
        body: JSON.stringify(body),
      });
    },
    deleteTransaction: (id: number) =>
      fetch(`${BASE}/api/portfolio/transactions/${id}`, { method: 'DELETE' }).then((r) => {
        if (!r.ok && r.status !== 204) throw new Error(`delete failed: ${r.status}`);
      }),
  },
  models: {
    screen: (
      params: {
        top?: number;
        days?: number;
        min_top?: number;
        experiment?: string;
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
        boards?: string[]; // serialized as comma list
        exclude_st?: boolean;
      } = {},
    ) => {
      type R = paths['/api/models/screen']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      const setNullable = (key: string, v: number | null | undefined) => {
        if (v !== undefined && v !== null) q.set(key, String(v));
      };
      if (params.top !== undefined) q.set('top', String(params.top));
      if (params.days !== undefined) q.set('days', String(params.days));
      if (params.min_top !== undefined) q.set('min_top', String(params.min_top));
      if (params.experiment) q.set('experiment', params.experiment);
      if (params.view) q.set('view', params.view);
      setNullable('min_price', params.min_price);
      setNullable('max_price', params.max_price);
      if (params.pct_change_n !== undefined) q.set('pct_change_n', String(params.pct_change_n));
      setNullable('min_pct_change', params.min_pct_change);
      setNullable('max_pct_change', params.max_pct_change);
      setNullable('min_amplitude', params.min_amplitude);
      setNullable('max_amplitude', params.max_amplitude);
      setNullable('min_vol_ratio', params.min_vol_ratio);
      setNullable('max_vol_ratio', params.max_vol_ratio);
      if (params.new_high_n !== undefined) q.set('new_high_n', String(params.new_high_n));
      if (params.boards && params.boards.length > 0) q.set('boards', params.boards.join(','));
      if (params.exclude_st !== undefined) q.set('exclude_st', params.exclude_st ? 'true' : 'false');
      const qs = q.toString();
      return request<R>(`/api/models/screen${qs ? '?' + qs : ''}`);
    },
    candidates: (
      params: {
        top?: number;
        days?: number;
        min_top?: number;
        experiment?: string;
        view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
        models?: string[];
      } = {},
    ) => {
      type R = paths['/api/models/candidates']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      if (params.top !== undefined) q.set('top', String(params.top));
      if (params.days !== undefined) q.set('days', String(params.days));
      if (params.min_top !== undefined) q.set('min_top', String(params.min_top));
      if (params.experiment) q.set('experiment', params.experiment);
      if (params.view) q.set('view', params.view);
      if (params.models && params.models.length > 0) q.set('models', params.models.join(','));
      const qs = q.toString();
      return request<R>(`/api/models/candidates${qs ? '?' + qs : ''}`);
    },
    predictions: (
      symbol: string,
      params: {
        days?: number;
        experiment?: string;
        view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
      } = {},
    ) => {
      type R = paths['/api/models/predictions/{symbol}']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      if (params.days !== undefined) q.set('days', String(params.days));
      if (params.experiment) q.set('experiment', params.experiment);
      if (params.view) q.set('view', params.view);
      const qs = q.toString();
      return request<R>(`/api/models/predictions/${encodeURIComponent(symbol)}${qs ? '?' + qs : ''}`);
    },
    experiments: () => {
      type R = paths['/api/models/experiments']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/models/experiments');
    },
    version: () => {
      type R = paths['/api/models/version']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/models/version');
    },
    rollback: (target: 'previous_1' | 'previous_2' = 'previous_1') => {
      type R =
        paths['/api/models/rollback']['post']['responses']['200']['content']['application/json'];
      return request<R>('/api/models/rollback', {
        method: 'POST',
        body: JSON.stringify({ target }),
      });
    },
  },
  evaluation: {
    listRecorders: () => {
      type R = paths['/api/evaluation/recorders']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/evaluation/recorders');
    },
    run: (
      body: paths['/api/evaluation/run']['post']['requestBody']['content']['application/json'],
    ) => {
      type R = paths['/api/evaluation/run']['post']['responses']['200']['content']['application/json'];
      return request<R>('/api/evaluation/run', { method: 'POST', body: JSON.stringify(body) });
    },
    getResult: (recorderId: string) => {
      type R = paths['/api/evaluation/results/{recorder_id}']['get']['responses']['200']['content']['application/json'];
      return request<R>(`/api/evaluation/results/${encodeURIComponent(recorderId)}`);
    },
    compare: (a: string, b: string, opts: { top_k?: number; cost_bps?: number } = {}) => {
      type R = paths['/api/evaluation/compare']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams({ a, b });
      if (opts.top_k !== undefined) q.set('top_k', String(opts.top_k));
      if (opts.cost_bps !== undefined) q.set('cost_bps', String(opts.cost_bps));
      return request<R>(`/api/evaluation/compare?${q.toString()}`);
    },
  },
  inference: {
    active: () =>
      request<{
        job_id: string;
        status: 'running' | 'done' | 'failed';
        started_at: string;
        finished_at: string | null;
        end_date: string | null;
        error: string | null;
        new_rows: number | null;
        reason: string | null;
      } | null>('/api/inference/active/peek'),
    status: () =>
      request<{
        last_run_at: string | null;
        last_success_at: string | null;
        last_error: string | null;
        is_running: boolean;
      }>('/api/inference/status'),
    runNow: (force = false) => {
      const q = new URLSearchParams({ force: String(force) });
      return request<{ status: string; job_id: string | null }>(
        `/api/inference/run-now?${q.toString()}`,
        { method: 'POST' },
      );
    },
    getJob: (jobId: string) =>
      request<{
        job_id: string;
        status: 'running' | 'done' | 'failed';
        started_at: string;
        finished_at: string | null;
        end_date: string | null;
        error: string | null;
        new_rows: number | null;
        reason: string | null;
      }>(`/api/inference/jobs/${encodeURIComponent(jobId)}`),
  },
  scheduling: {
    getRetrain: () => {
      type R = paths['/api/scheduling/retrain']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/scheduling/retrain');
    },
    putRetrain: (
      body: paths['/api/scheduling/retrain']['put']['requestBody']['content']['application/json'],
    ) => {
      type R = paths['/api/scheduling/retrain']['put']['responses']['200']['content']['application/json'];
      return request<R>('/api/scheduling/retrain', {
        method: 'PUT',
        body: JSON.stringify(body),
      });
    },
    runNow: (force = false) => {
      type R =
        paths['/api/scheduling/retrain/run-now']['post']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams({ force: String(force) });
      return request<R>(`/api/scheduling/retrain/run-now?${q.toString()}`, { method: 'POST' });
    },
  },
};
