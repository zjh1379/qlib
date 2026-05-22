import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';

import { DEFAULT_FILTERS, FilterParams } from './types';
import { paramsFromUrl, urlFromParams } from './parse';

/** Two-way bind FilterParams ↔ URL query string. Reading is O(1), writing
 *  replaces the URL via react-router's `setSearchParams` (no full reload). */
export function useFilterParams(): [FilterParams, (next: Partial<FilterParams>) => void, () => void] {
  const [sp, setSp] = useSearchParams();
  const params = useMemo(() => paramsFromUrl(sp), [sp]);

  const update = useCallback(
    (patch: Partial<FilterParams>) => {
      const merged: FilterParams = { ...params, ...patch };
      const nextSp = urlFromParams(merged);
      setSp(nextSp, { replace: false });
    },
    [params, setSp],
  );

  const reset = useCallback(() => {
    setSp(urlFromParams(DEFAULT_FILTERS), { replace: false });
  }, [setSp]);

  return [params, update, reset];
}
