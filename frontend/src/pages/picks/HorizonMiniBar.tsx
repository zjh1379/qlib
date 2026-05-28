import { cn } from '@/lib/utils';

interface Props {
  horizon: '1d' | '5d' | '20d';
  predReturn: number | null;
  percentile: number;
  modelAgreement: number | null;
  /** Max abs return across visible rows for this column, used to normalize
   * bar widths so the eye can compare magnitudes across stocks. If absent,
   * uses a single-row baseline of 5%. */
  maxAbsReturn?: number;
  isPrimary?: boolean;
}

const HORIZON_LABEL: Record<string, string> = {
  '1d': '次日',
  '5d': '5 日',
  '20d': '20 日',
};

export default function HorizonMiniBar({
  horizon,
  predReturn,
  percentile,
  modelAgreement,
  maxAbsReturn,
  isPrimary,
}: Props) {
  const isPositive = predReturn != null && predReturn > 0;
  const isNegative = predReturn != null && predReturn < 0;

  // A-share convention: red = up, green = down. Match PredictionChart colors.
  const baseClasses = isPositive
    ? 'bg-red-500/40 border-red-500/70 text-red-50'
    : isNegative
      ? 'bg-green-500/40 border-green-500/70 text-green-50'
      : 'bg-gray-700/40 border-gray-700/60 text-gray-300';

  const widthPct = predReturn != null
    ? Math.min(100, (Math.abs(predReturn) / (maxAbsReturn ?? 0.05)) * 100)
    : 0;

  const topPct = Math.max(0, 100 - percentile);
  const topLabel = topPct < 0.1 ? 'top 0.1%' : `top ${topPct.toFixed(1)}%`;

  const showStar = modelAgreement != null && modelAgreement >= 0.99;

  const returnLabel = predReturn != null
    ? `${predReturn >= 0 ? '+' : ''}${(predReturn * 100).toFixed(1)}%`
    : null;

  return (
    <div
      className={cn(
        'flex flex-col gap-0.5 min-w-[78px]',
        isPrimary && 'ring-1 ring-blue-500/40 pl-1 pr-1 rounded',
      )}
      title={`${HORIZON_LABEL[horizon]} 预期收益 ${returnLabel ?? 'N/A'} · 排名 ${topLabel}`}
    >
      <div
        data-testid="mini-bar"
        className={cn(
          'h-4 rounded-sm border flex items-center justify-end pr-1',
          baseClasses,
        )}
        style={{ width: `${Math.max(20, widthPct)}%`, minWidth: '40px' }}
      >
        {returnLabel && (
          <span className="text-[10px] font-medium">{returnLabel}</span>
        )}
      </div>
      <div className="text-[10px] text-[#8b949e] flex items-center gap-1">
        <span>{topLabel}</span>
        {showStar && (
          <span className="text-yellow-400" title="3 个模型方向一致">★</span>
        )}
      </div>
    </div>
  );
}
