import { Link } from 'react-router-dom';
import type { components } from '@/api/types.gen';
import { cn } from '@/lib/utils';

type Holding = components['schemas']['Holding'];
type HoldingsResponse = components['schemas']['HoldingsResponse'];

interface Props {
  data: HoldingsResponse;
}

function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return n.toLocaleString('zh-CN', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPct(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  const sign = n >= 0 ? '+' : '';
  return `${sign}${(n * 100).toFixed(2)}%`;
}

function pnlClass(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return 'text-[#8b949e]';
  if (n > 0) return 'text-green-400';
  if (n < 0) return 'text-red-400';
  return 'text-[#8b949e]';
}

export default function HoldingsTable({ data }: Props) {
  const holdings = data.holdings;
  if (holdings.length === 0) {
    return (
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-8 text-center">
        <p className="text-sm text-[#8b949e]">
          暂无持仓，点击右上角"+ 添加交易"录入第一笔
        </p>
      </div>
    );
  }

  const totalCost = data.total_cost;
  const totalMv = data.total_market_value ?? null;
  const totalPnl = data.total_unrealized_pnl ?? null;
  const totalPnlPct = totalCost > 0 && totalPnl !== null ? totalPnl / totalCost : null;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-[#161b22] text-[#8b949e]">
            <tr>
              <Th>代码</Th>
              <Th>名称</Th>
              <Th align="right">数量</Th>
              <Th align="right">平均成本</Th>
              <Th align="right">现价</Th>
              <Th align="right">市值</Th>
              <Th align="right">浮动盈亏</Th>
              <Th align="right">涨跌幅</Th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((h: Holding) => (
              <tr
                key={h.symbol}
                className="border-t border-[#21262d] hover:bg-[#161b22] transition cursor-pointer"
              >
                <Td>
                  <Link
                    to={`/charts/${h.symbol}`}
                    className="font-mono text-[#58a6ff] hover:underline"
                  >
                    {h.symbol}
                  </Link>
                </Td>
                <Td>{h.name || '—'}</Td>
                <Td align="right" mono>
                  {fmtNum(h.qty, 0)}
                </Td>
                <Td align="right" mono>
                  {fmtNum(h.avg_cost, 3)}
                </Td>
                <Td align="right" mono>
                  {fmtNum(h.current_price)}
                </Td>
                <Td align="right" mono>
                  {fmtNum(h.market_value)}
                </Td>
                <Td align="right" mono className={pnlClass(h.unrealized_pnl)}>
                  {h.unrealized_pnl !== null && h.unrealized_pnl !== undefined
                    ? (h.unrealized_pnl >= 0 ? '+' : '') + fmtNum(h.unrealized_pnl)
                    : '—'}
                </Td>
                <Td align="right" mono className={pnlClass(h.unrealized_pnl_pct)}>
                  {fmtPct(h.unrealized_pnl_pct)}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h3 className="text-xs font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          汇总
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <Stat label="总投入" value={fmtNum(totalCost)} />
          <Stat label="总市值" value={fmtNum(totalMv)} />
          <Stat
            label="总浮盈"
            value={
              <span className={cn('font-mono', pnlClass(totalPnl))}>
                {totalPnl !== null && totalPnl !== undefined
                  ? (totalPnl >= 0 ? '+' : '') + fmtNum(totalPnl)
                  : '—'}
              </span>
            }
          />
          <Stat
            label="总收益率"
            value={
              <span className={cn('font-mono', pnlClass(totalPnlPct))}>
                {fmtPct(totalPnlPct)}
              </span>
            }
          />
        </div>
        {data.as_of && (
          <div className="text-xs text-[#6e7681] mt-3">
            截至: {new Date(data.as_of).toLocaleString('zh-CN')}
          </div>
        )}
      </div>
    </div>
  );
}

function Th({
  children,
  align = 'left',
}: {
  children: React.ReactNode;
  align?: 'left' | 'right';
}) {
  return (
    <th
      className={cn(
        'px-4 py-2 text-xs font-medium uppercase tracking-wider',
        align === 'right' ? 'text-right' : 'text-left',
      )}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = 'left',
  mono = false,
  className,
}: {
  children: React.ReactNode;
  align?: 'left' | 'right';
  mono?: boolean;
  className?: string;
}) {
  return (
    <td
      className={cn(
        'px-4 py-2.5',
        align === 'right' ? 'text-right' : 'text-left',
        mono && 'font-mono',
        className,
      )}
    >
      {children}
    </td>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</div>
      <div className="text-base mt-1 font-mono">{value}</div>
    </div>
  );
}
