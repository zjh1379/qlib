import type { components } from '@/api/types.gen';
import type { Board, FilterParams } from './types';

type Candidate = components['schemas']['ScreenItem'];

/** Apply Tier 1 filters to the candidate pool client-side.
 *  AND semantics across filter groups; OR within boards multi-select.
 *  Symbols missing a metric fail any non-trivial bound on that metric.
 *  Mirrors the server-side apply_tier1_filters logic. */
export function applyFilters(candidates: Candidate[], filters: FilterParams): Candidate[] {
  return candidates.filter((c) => {
    // Price range
    if (!passesRange(c.last_price, filters.min_price, filters.max_price)) return false;

    // pct_change for the user-selected N-day window
    const pctKey = `pct_change_${filters.pct_change_n}d` as keyof Candidate;
    const pct = (c[pctKey] as number | null | undefined) ?? null;
    if (!passesRange(pct, filters.min_pct_change, filters.max_pct_change)) return false;

    // Amplitude
    if (!passesRange(c.amplitude, filters.min_amplitude, filters.max_amplitude)) return false;

    // Volume ratio
    if (!passesRange(c.vol_ratio, filters.min_vol_ratio, filters.max_vol_ratio)) return false;

    // New high
    if (filters.new_high_n !== 0) {
      const key = `is_new_high_${filters.new_high_n}d` as keyof Candidate;
      if (!c[key]) return false;
    }

    // Boards (OR within selected)
    if (filters.boards.length > 0) {
      const cb = (c.board ?? '') as Board;
      if (!filters.boards.includes(cb)) return false;
    }

    // ST exclusion
    if (filters.exclude_st && c.is_st) return false;

    // Consensus (UI-only filter)
    if ((c.consensus ?? 0) < filters.min_consensus) return false;

    return true;
  });
}

function passesRange(value: number | null | undefined, lo: number | null, hi: number | null): boolean {
  if (lo === null && hi === null) return true;
  if (value === null || value === undefined) return false;
  if (lo !== null && value < lo) return false;
  if (hi !== null && value > hi) return false;
  return true;
}
