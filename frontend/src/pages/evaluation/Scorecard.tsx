import { cn } from '@/lib/utils';
import type { ScorecardData } from './types';
import { ACCEPTANCE_THRESHOLDS } from './types';

interface ScorecardProps {
  data: ScorecardData;
}

interface Row {
  label: string;
  value: string;
  threshold: string;
  ok: boolean;
  hint?: string;
}

export function Scorecard({ data }: ScorecardProps) {
  const rows: Row[] = [
    {
      label: 'IC mean',
      value: fmt(data.ic_mean, 4, true),
      threshold: `≥ ${ACCEPTANCE_THRESHOLDS.ic_mean.toFixed(3)}`,
      ok: data.ic_mean >= ACCEPTANCE_THRESHOLDS.ic_mean,
      hint: '日横截面 Pearson 相关；信号纯度核心',
    },
    {
      label: 'RIC mean',
      value: fmt(data.ric_mean, 4, true),
      threshold: '—',
      ok: true,
      hint: 'Spearman 相关；rank-based',
    },
    {
      label: 'ICIR',
      value: fmt(data.icir, 4, true),
      threshold: '≥ 0.40',
      ok: data.icir >= 0.40,
      hint: '日 IC 的均值/标准差；衡量稳定性',
    },
    {
      label: 'Top-Bottom Spread',
      value: `${fmt(data.top_bottom_spread_monthly, 2, true)}%/月`,
      threshold: '≥ 1.5%/月',
      ok: data.top_bottom_spread_monthly >= 1.5,
      hint: '前30 - 后30 月度平均涨幅差',
    },
    {
      label: '年化超额收益',
      value: `${fmt(data.annual_excess_return * 100, 2, true)}%`,
      threshold: '≥ +15%',
      ok: data.annual_excess_return >= 0.15,
      hint: 'Top30 长仓年化收益（已扣交易成本）',
    },
    {
      label: 'IR (cost-adj)',
      value: fmt(data.ir, 4, true),
      threshold: `≥ ${ACCEPTANCE_THRESHOLDS.ir.toFixed(1)}`,
      ok: data.ir >= ACCEPTANCE_THRESHOLDS.ir,
      hint: '年化收益 / 年化波动；风险调整后表现',
    },
    {
      label: 'Max Drawdown',
      value: `${fmt(data.max_drawdown * 100, 2, true)}%`,
      threshold: `≥ ${(ACCEPTANCE_THRESHOLDS.max_drawdown * 100).toFixed(0)}%`,
      ok: data.max_drawdown >= ACCEPTANCE_THRESHOLDS.max_drawdown,
      hint: '组合的最大回撤；越接近 0 越好',
    },
    {
      label: '日均换手',
      value: `${fmt(data.daily_turnover * 100, 2)}%`,
      threshold: `≤ ${(ACCEPTANCE_THRESHOLDS.daily_turnover * 100).toFixed(0)}%`,
      ok: data.daily_turnover <= ACCEPTANCE_THRESHOLDS.daily_turnover,
      hint: '组合每日换出比例；过高 = 实盘吃手续费',
    },
  ];

  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
        8 指标记分卡
      </h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
            <th className="py-2 pr-4">指标</th>
            <th className="py-2 pr-4 text-right">取值</th>
            <th className="py-2 pr-4 text-right">阈值</th>
            <th className="py-2 pr-4 text-center">通过</th>
            <th className="py-2 pr-4 text-[#6e7681]">说明</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-[#21262d]">
              <td className="py-2 pr-4 text-[#e6edf3]">{r.label}</td>
              <td className={cn('py-2 pr-4 text-right font-mono', r.ok ? 'text-green-400' : 'text-red-400')}>
                {r.value}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-[#8b949e]">{r.threshold}</td>
              <td className="py-2 pr-4 text-center">
                {r.threshold === '—' ? (
                  <span className="text-[#6e7681]">—</span>
                ) : r.ok ? (
                  <span className="text-green-400">✓</span>
                ) : (
                  <span className="text-red-400">✗</span>
                )}
              </td>
              <td className="py-2 pr-4 text-xs text-[#6e7681]">{r.hint}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function fmt(v: number, decimals: number, signed: boolean = false): string {
  if (!Number.isFinite(v)) return 'NaN';
  const s = v.toFixed(decimals);
  return signed && v >= 0 ? '+' + s : s;
}
