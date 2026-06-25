import type { components } from '@/api/types.gen';
import type { Board, FilterParams } from './types';
import { daysInTop, windowScoreAvg } from './persistence';

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

/** Full client-side pipeline producing the displayed rows:
 *  1) attribute filters (price/trend/board/ST/consensus) via applyFilters
 *  2) window score_avg recompute (over last D days)
 *  3) persistence filter: days in top-`top` over last D days >= min_top
 *  4) canonical rank by window score_avg desc
 *  5) display cap to `top` rows
 *  Returns the selected, rank-assigned rows (display sort applied by caller). */
export function selectCandidates(
  candidates: Candidate[],
  filters: FilterParams,
  windowDatesLen: number,
): Candidate[] {
  const D = Math.max(1, Math.min(filters.days, windowDatesLen || filters.days));
  const passed = applyFilters(candidates, filters)
    .map((c) => ({
      ...c,
      score_avg: windowScoreAvg(c.daily_scores ?? [], D) ?? c.score_avg,
    }))
    .filter((c) => daysInTop(c.daily_ranks ?? [], D, filters.top) >= filters.min_top);
  passed.sort((a, b) => (b.score_avg ?? -Infinity) - (a.score_avg ?? -Infinity));
  return passed
    .map((c, i) => ({ ...c, rank: i + 1 }))
    .slice(0, filters.top);
}
