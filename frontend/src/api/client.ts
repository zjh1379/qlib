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
    screen: (params: { top?: number; days?: number; min_top?: number; experiment?: string } = {}) => {
      type R = paths['/api/models/screen']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      if (params.top !== undefined) q.set('top', String(params.top));
      if (params.days !== undefined) q.set('days', String(params.days));
      if (params.min_top !== undefined) q.set('min_top', String(params.min_top));
      if (params.experiment) q.set('experiment', params.experiment);
      const qs = q.toString();
      return request<R>(`/api/models/screen${qs ? '?' + qs : ''}`);
    },
    predictions: (symbol: string, params: { days?: number; experiment?: string } = {}) => {
      type R = paths['/api/models/predictions/{symbol}']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      if (params.days !== undefined) q.set('days', String(params.days));
      if (params.experiment) q.set('experiment', params.experiment);
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
