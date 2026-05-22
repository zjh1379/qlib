export type View = 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
export type Board = 'main' | 'gem' | 'star' | 'bj' | 'etf';
export type PctChangeN = 1 | 3 | 5 | 10 | 20;
export type NewHighN = 0 | 20 | 60 | 120;

export const PCT_CHANGE_N_OPTIONS: PctChangeN[] = [1, 3, 5, 10, 20];
export const NEW_HIGH_N_OPTIONS: NewHighN[] = [0, 20, 60, 120];

export const BOARDS: { value: Board; label: string }[] = [
  { value: 'main', label: '主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bj', label: '北交所' },
  { value: 'etf', label: 'ETF' },
];

export interface FilterParams {
  // Core (existing)
  top: number;
  days: number;
  min_top: number;
  view: View;
  min_price: number | null;
  max_price: number | null;
  // Tier 1
  pct_change_n: PctChangeN;
  min_pct_change: number | null;
  max_pct_change: number | null;
  min_amplitude: number | null;
  max_amplitude: number | null;
  min_vol_ratio: number | null;
  max_vol_ratio: number | null;
  new_high_n: NewHighN;
  boards: Board[];   // empty = no filter
  exclude_st: boolean;
  // UI-only (not sent to backend)
  min_consensus: number;
}

export const DEFAULT_FILTERS: FilterParams = {
  top: 30,
  days: 5,
  min_top: 0,
  view: 'ensemble',
  min_price: null,
  max_price: 30,
  pct_change_n: 5,
  min_pct_change: null,
  max_pct_change: null,
  min_amplitude: null,
  max_amplitude: null,
  min_vol_ratio: null,
  max_vol_ratio: null,
  new_high_n: 0,
  boards: [],
  exclude_st: true,
  min_consensus: 0,
};
