// frontend/src/pages/picks/useRecompute.ts
import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { components } from '@/api/types.gen';
import { comboKey } from './persistence';

type RecomputeJob = components['schemas']['RecomputeJob'];

export interface RecomputeController {
  /** combos warmed this session (backend lru_cache hot + GET allowed). */
  isWarmed: (view: string, models: string[]) => boolean;
  /** start a recompute for a combo; progress surfaces via `job`. */
  start: (view: string, models: string[]) => Promise<void>;
  /** latest polled job (running/done/failed) or null. */
  job: RecomputeJob | null;
  /** elapsed seconds since the active job started (0 when idle). */
  elapsedSec: number;
}

export function useRecompute(onWarmed?: (view: string, models: string[]) => void): RecomputeController {
  const [warmed, setWarmed] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const pending = useRef<{ view: string; models: string[] } | null>(null);
  const onWarmedRef = useRef(onWarmed);
  onWarmedRef.current = onWarmed;

  const { data: job } = useQuery({
    queryKey: ['recompute', jobId],
    queryFn: () => api.models.recomputeStatus(jobId as string),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 800 : false),
  });

  // elapsed timer while a job is running
  useEffect(() => {
    if (!startedAt) return;
    const h = setInterval(() => setElapsedSec(Math.round((Date.now() - startedAt) / 1000)), 250);
    return () => clearInterval(h);
  }, [startedAt]);

  // when job finishes, mark warmed (on done) and stop the timer
  useEffect(() => {
    if (!job) return;
    if (job.status === 'done' && pending.current) {
      const { view, models } = pending.current;
      setWarmed((prev) => new Set(prev).add(comboKey(view, models)));
      onWarmedRef.current?.(view, models);
    }
    if (job.status === 'done' || job.status === 'failed') {
      setStartedAt(null);
      pending.current = null;
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const isWarmed = useCallback(
    (view: string, models: string[]) => warmed.has(comboKey(view, models)),
    [warmed],
  );

  const start = useCallback(async (view: string, models: string[]) => {
    pending.current = { view, models };
    setStartedAt(Date.now());
    setElapsedSec(0);
    const res = await api.models.recompute({ view, models });
    if (res.job_id) setJobId(res.job_id);
  }, []);

  return { isWarmed, start, job: job ?? null, elapsedSec };
}
