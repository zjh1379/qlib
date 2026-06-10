import { useQuery } from '@tanstack/react-query';
import type { ActiveJobKind } from './useActiveJobs';

/**
 * Poll one background-job "active" endpoint with the shared adaptive cadence.
 *
 * This is the single home for the job-polling policy: queryKey shape
 * (`['jobs', kind, 'active']`), background refetching, and staleTime tied to the
 * interval. The caller owns the interval (so all jobs flip fast↔slow together
 * off one `hasActive` flag) and passes it in. Returns the latest data (or
 * undefined while loading) — callers already handle undefined.
 */
export function useJobPolling<T>(
  kind: ActiveJobKind,
  queryFn: () => Promise<T>,
  intervalMs: number,
): T | undefined {
  return useQuery({
    queryKey: ['jobs', kind, 'active'],
    queryFn,
    staleTime: intervalMs / 2,
    refetchInterval: intervalMs,
    refetchIntervalInBackground: true,
  }).data;
}
