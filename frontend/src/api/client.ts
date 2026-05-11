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
};
