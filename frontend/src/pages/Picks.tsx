import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useScreen } from '@/models/hooks';
import { cn } from '@/lib/utils';

type View = 'ensemble' | 'lightgbm' | 'alstm' | 'tra';

const VIEW_OPTIONS: { value: View; label: string }[] = [
  { value: 'ensemble', label: '集成 (Ensemble)' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'alstm', label: 'ALSTM' },
  { value: 'tra', label: 'TRA' },
];

export default function Picks() {
  const [top, setTop] = useState(30);
  const [days, setDays] = useState(5);
  const [minTop, setMinTop] = useState(0);
  const [view, setView] = useState<View>('ensemble');
  const [minConsensus, setMinConsensus] = useState(0);

  const { data, isPending, error } = useScreen({ top, days, min_top: minTop, view });

  const filteredItems = data
    ? data.items.filter((it) => (it.consensus ?? 0) >= minConsensus)
    : [];

  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold">选股工作台</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          基于滚动重训集成模型的横截面打分排名 · 可切换视图查看单模型分
        </p>
      </header>

      {/* Filter bar */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          筛选
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <ViewSelect value={view} onChange={setView} />
          <NumberInput label="Top N" value={top} onChange={setTop} min={1} max={300} />
          <NumberInput label="窗口天数" value={days} onChange={setDays} min={1} max={60} />
          <NumberInput
            label="最少进 top N 天数"
            value={minTop}
            onChange={setMinTop}
            min={0}
            max={days}
          />
          <ConsensusSlider value={minConsensus} onChange={setMinConsensus} />
        </div>
      </div>

      {/* Results table */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          结果 {data ? `(${filteredItems.length}/${data.items.length})` : ''}
        </h2>
        {error ? (
          <div className="text-red-400 text-sm">加载失败: {(error as Error).message}</div>
        ) : isPending ? (
          <div className="text-[#8b949e] text-sm">加载中…</div>
        ) : data && filteredItems.length === 0 ? (
          <div className="text-[#8b949e] text-sm">没有符合条件的股票。</div>
        ) : data ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
                  <th className="py-2 pr-4">rank</th>
                  <th className="py-2 pr-4">代码</th>
                  <th className="py-2 pr-4">名称</th>
                  <th className="py-2 pr-4 text-right">score_today</th>
                  <th className="py-2 pr-4 text-right">score_avg</th>
                  <th className="py-2 pr-4 text-right">rank_avg</th>
                  <th className="py-2 pr-4 text-right">days_in_top</th>
                  <th className="py-2 pr-4 text-right">共识</th>
                </tr>
              </thead>
              <tbody>
                {filteredItems.map((item) => (
                  <tr
                    key={item.symbol}
                    className="border-b border-[#21262d] hover:bg-[#161b22] transition cursor-pointer"
                  >
                    <td className="py-2 pr-4 font-mono text-[#8b949e]">{item.rank}</td>
                    <td className="py-2 pr-4">
                      <Link
                        to={`/charts/${item.symbol}`}
                        className="font-mono text-[#58a6ff] hover:underline"
                      >
                        {item.symbol}
                      </Link>
                    </td>
                    <td className="py-2 pr-4">
                      <Link to={`/charts/${item.symbol}`} className="hover:underline">
                        {item.name}
                      </Link>
                    </td>
                    <td
                      className={cn(
                        'py-2 pr-4 text-right font-mono',
                        scoreColorClass(item.score_today),
                      )}
                    >
                      {formatScore(item.score_today)}
                    </td>
                    <td
                      className={cn(
                        'py-2 pr-4 text-right font-mono',
                        scoreColorClass(item.score_avg),
                      )}
                    >
                      {formatScore(item.score_avg)}
                    </td>
                    <td
                      className={cn(
                        'py-2 pr-4 text-right font-mono',
                        rankColorClass(item.rank_avg),
                      )}
                    >
                      {item.rank_avg.toFixed(1)}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono">{item.days_in_top}</td>
                    <td
                      className={cn(
                        'py-2 pr-4 text-right font-mono',
                        consensusColorClass(item.consensus ?? 0),
                      )}
                    >
                      {(item.consensus ?? 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>

      {/* Footer info */}
      {data && (
        <div className="text-xs text-[#6e7681] grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span className="uppercase tracking-wider">experiment</span>
            <div className="font-mono text-[#8b949e] mt-1">{data.experiment}</div>
          </div>
          <div>
            <span className="uppercase tracking-wider">recorder_id</span>
            <div className="font-mono text-[#8b949e] mt-1 truncate">{data.recorder_id}</div>
          </div>
          <div>
            <span className="uppercase tracking-wider">latest_date</span>
            <div className="font-mono text-[#8b949e] mt-1">{data.latest_date}</div>
          </div>
          <div>
            <span className="uppercase tracking-wider">universe_size</span>
            <div className="font-mono text-[#8b949e] mt-1">
              {data.universe_size.toLocaleString()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
}) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) =>
          onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))
        }
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      />
    </label>
  );
}

function ViewSelect({ value, onChange }: { value: View; onChange: (v: View) => void }) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">视图</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as View)}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      >
        {VIEW_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ConsensusSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">
        最低共识 ({value.toFixed(2)})
      </span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full h-9 accent-[#1f6feb]"
      />
    </label>
  );
}

function formatScore(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(4);
}

function scoreColorClass(v: number): string {
  if (v > 0.0005) return 'text-green-400';
  if (v < -0.0005) return 'text-red-400';
  return 'text-[#8b949e]';
}

function rankColorClass(rank: number): string {
  // Lower rank = better. Highlight top ranks in green, weaken as it grows.
  if (rank <= 20) return 'text-green-400';
  if (rank <= 50) return 'text-[#e6edf3]';
  return 'text-[#8b949e]';
}

function consensusColorClass(v: number): string {
  // Higher consensus = more models agree on direction. Spec thresholds:
  // >= 0.78 green, >= 0.44 yellow, < 0.44 gray.
  if (v >= 0.78) return 'text-green-400';
  if (v >= 0.44) return 'text-yellow-400';
  return 'text-[#8b949e]';
}
