import type { Board, FilterParams, NewHighN, PctChangeN, View } from './types';
import { BOARDS, DEFAULT_FILTERS, NEW_HIGH_N_OPTIONS, PCT_CHANGE_N_OPTIONS } from './types';

const BOARD_VALUES: readonly Board[] = BOARDS.map((b) => b.value);
const VIEWS: readonly View[] = ['ensemble', 'lightgbm', 'alstm', 'tra'];

export function parseInt32(raw: string | null, fallback: number): number {
  if (raw === null || raw === '') return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

export function parseFloat32(raw: string | null, fallback: number | null): number | null {
  if (raw === null || raw === '') return fallback;
  const n = Number.parseFloat(raw);
  return Number.isFinite(n) ? n : fallback;
}

export function parseBool(raw: string | null, fallback: boolean): boolean {
  if (raw === null) return fallback;
  return raw === 'true' || raw === '1';
}

export function parseEnum<T extends string | number>(
  raw: string | null,
  allowed: readonly T[],
  fallback: T,
): T {
  if (raw === null || raw === '') return fallback;
  const parsed: string | number = typeof allowed[0] === 'number' ? Number(raw) : raw;
  return (allowed as readonly (string | number)[]).includes(parsed) ? (parsed as T) : fallback;
}

export function parseBoards(raw: string | null): Board[] {
  if (raw === null || raw === '') return [];
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s): s is Board => (BOARD_VALUES as readonly string[]).includes(s));
}

export function serializeBoards(boards: Board[]): string {
  return boards.join(',');
}

/** Parse a URLSearchParams instance into FilterParams, falling back to DEFAULT_FILTERS for any missing key. */
export function paramsFromUrl(sp: URLSearchParams): FilterParams {
  return {
    top: parseInt32(sp.get('top'), DEFAULT_FILTERS.top),
    days: parseInt32(sp.get('days'), DEFAULT_FILTERS.days),
    min_top: parseInt32(sp.get('min_top'), DEFAULT_FILTERS.min_top),
    view: parseEnum(sp.get('view'), VIEWS, DEFAULT_FILTERS.view),
    min_price: parseFloat32(sp.get('min_price'), DEFAULT_FILTERS.min_price),
    max_price: parseFloat32(sp.get('max_price'), DEFAULT_FILTERS.max_price),
    pct_change_n: parseEnum<PctChangeN>(
      sp.get('pct_change_n'),
      PCT_CHANGE_N_OPTIONS,
      DEFAULT_FILTERS.pct_change_n,
    ),
    min_pct_change: parseFloat32(sp.get('min_pct_change'), DEFAULT_FILTERS.min_pct_change),
    max_pct_change: parseFloat32(sp.get('max_pct_change'), DEFAULT_FILTERS.max_pct_change),
    min_amplitude: parseFloat32(sp.get('min_amplitude'), DEFAULT_FILTERS.min_amplitude),
    max_amplitude: parseFloat32(sp.get('max_amplitude'), DEFAULT_FILTERS.max_amplitude),
    min_vol_ratio: parseFloat32(sp.get('min_vol_ratio'), DEFAULT_FILTERS.min_vol_ratio),
    max_vol_ratio: parseFloat32(sp.get('max_vol_ratio'), DEFAULT_FILTERS.max_vol_ratio),
    new_high_n: parseEnum<NewHighN>(
      sp.get('new_high_n'),
      NEW_HIGH_N_OPTIONS,
      DEFAULT_FILTERS.new_high_n,
    ),
    boards: parseBoards(sp.get('boards')),
    exclude_st: parseBool(sp.get('exclude_st'), DEFAULT_FILTERS.exclude_st),
    min_consensus: parseFloat32(sp.get('min_consensus'), DEFAULT_FILTERS.min_consensus) ?? 0,
  };
}

/** Inverse: write only non-default keys back into the URL. */
export function urlFromParams(p: FilterParams): URLSearchParams {
  const sp = new URLSearchParams();
  const setIfChanged = (key: keyof FilterParams, val: unknown) => {
    if (val === null || val === undefined) return;
    if (val === DEFAULT_FILTERS[key]) return;
    if (Array.isArray(val)) {
      if (val.length === 0) return;
      sp.set(key, val.join(','));
    } else if (typeof val === 'boolean') {
      sp.set(key, val ? 'true' : 'false');
    } else {
      sp.set(key, String(val));
    }
  };
  setIfChanged('top', p.top);
  setIfChanged('days', p.days);
  setIfChanged('min_top', p.min_top);
  setIfChanged('view', p.view);
  setIfChanged('min_price', p.min_price);
  setIfChanged('max_price', p.max_price);
  setIfChanged('pct_change_n', p.pct_change_n);
  setIfChanged('min_pct_change', p.min_pct_change);
  setIfChanged('max_pct_change', p.max_pct_change);
  setIfChanged('min_amplitude', p.min_amplitude);
  setIfChanged('max_amplitude', p.max_amplitude);
  setIfChanged('min_vol_ratio', p.min_vol_ratio);
  setIfChanged('max_vol_ratio', p.max_vol_ratio);
  setIfChanged('new_high_n', p.new_high_n);
  setIfChanged('boards', p.boards);
  setIfChanged('exclude_st', p.exclude_st);
  setIfChanged('min_consensus', p.min_consensus);
  return sp;
}
