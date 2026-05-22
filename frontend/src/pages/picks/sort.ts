import type { components } from '@/api/types.gen';

type Candidate = components['schemas']['ScreenItem'];

/** Sortable columns. Keep narrow: only the user-facing display columns. */
export type SortKey =
  | 'rank'
  | 'symbol'
  | 'last_price'
  | 'pct_change_5d'
  | 'amplitude'
  | 'vol_ratio'
  | 'consensus'
  | 'score_avg'
  | 'days_in_top';

export type SortDir = 'asc' | 'desc';

export interface SortState {
  key: SortKey;
  dir: SortDir;
}

export const DEFAULT_SORT: SortState = { key: 'rank', dir: 'asc' };

/** Stable, null-safe sort. Nulls go to the end regardless of direction. */
export function applySort(items: Candidate[], sort: SortState): Candidate[] {
  const { key, dir } = sort;
  const sign = dir === 'asc' ? 1 : -1;
  // [...items] avoids mutating the input (which React/TanStack may hold by reference).
  return [...items].sort((a, b) => {
    const av = a[key] as number | string | null | undefined;
    const bv = b[key] as number | string | null | undefined;
    const aNull = av === null || av === undefined;
    const bNull = bv === null || bv === undefined;
    if (aNull && bNull) return 0;
    if (aNull) return 1;   // null always last
    if (bNull) return -1;
    if (typeof av === 'number' && typeof bv === 'number') {
      return sign * (av - bv);
    }
    return sign * String(av).localeCompare(String(bv));
  });
}

/** Toggle behavior: clicking the active column flips direction; clicking a
 *  different column resets to descending (sensible default for score-like metrics). */
export function nextSort(current: SortState, clicked: SortKey): SortState {
  if (current.key === clicked) {
    return { key: clicked, dir: current.dir === 'asc' ? 'desc' : 'asc' };
  }
  // Different column: default to 'desc' for numeric metrics (highest first feels right
  // for prices, score, pct_change, consensus). Exception: rank is naturally ascending.
  const defaultDir: SortDir = clicked === 'rank' || clicked === 'symbol' ? 'asc' : 'desc';
  return { key: clicked, dir: defaultDir };
}
