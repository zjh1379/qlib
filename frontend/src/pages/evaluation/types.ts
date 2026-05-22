import type { components } from '@/api/types.gen';

export type RecorderSummary = components['schemas']['RecorderSummary'];
export type EvalResult = components['schemas']['EvalResult'];
export type ScorecardData = components['schemas']['ScorecardData'];
export type RegimeMetrics = components['schemas']['RegimeMetrics'];
export type AcceptanceResult = components['schemas']['AcceptanceResult'];
export type CompareResult = components['schemas']['CompareResult'];

/** Acceptance thresholds (mirror production/validate_acceptance.py THRESHOLDS). */
export const ACCEPTANCE_THRESHOLDS = {
  ic_mean: 0.030,
  ir: 2.5,
  max_drawdown: -0.15,
  daily_turnover: 0.20,
} as const;

/** Display label for each acceptance criterion. */
export const ACCEPTANCE_LABELS: Record<string, string> = {
  ic_mean: 'IC mean ≥ 0.030',
  ir: 'IR (cost-adj) ≥ 2.5',
  max_drawdown: 'Max DD ≤ 15%',
  daily_turnover: 'Daily turnover ≤ 20%',
  regimes_all_positive: 'All regime IRs > 0',
};
