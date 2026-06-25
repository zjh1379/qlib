// frontend/src/pages/picks/persistence.ts

/** Count days in the last `windowD` entries where rank is present and <= topN.
 *  Ranks are aligned ascending to CandidatesResponse.window_dates. */
export function daysInTop(
  dailyRanks: (number | null)[],
  windowD: number,
  topN: number,
): number {
  const slice = windowD >= dailyRanks.length ? dailyRanks : dailyRanks.slice(-windowD);
  let n = 0;
  for (const r of slice) if (r != null && r <= topN) n++;
  return n;
}

/** Mean of the last `windowD` non-null daily scores, or null if none. */
export function windowScoreAvg(
  dailyScores: (number | null)[],
  windowD: number,
): number | null {
  const slice = windowD >= dailyScores.length ? dailyScores : dailyScores.slice(-windowD);
  let sum = 0;
  let count = 0;
  for (const s of slice) if (s != null) { sum += s; count++; }
  return count === 0 ? null : sum / count;
}

/** Stable, order-independent key for a (view, models) combo. */
export function comboKey(view: string, models: string[]): string {
  return `${view}|${[...models].sort().join(',')}`;
}
