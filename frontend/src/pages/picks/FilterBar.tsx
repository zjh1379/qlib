import { useEffect, useState } from 'react';
import type { Board, FilterParams, NewHighN, PctChangeN, View } from './types';
import { BOARDS, NEW_HIGH_N_OPTIONS, PCT_CHANGE_N_OPTIONS } from './types';

interface FilterBarProps {
  params: FilterParams;
  resultCount: number | null;
  candidateCount: number | null;
  onChange: (patch: Partial<FilterParams>) => void;
  onReset: () => void;
}

const VIEW_OPTIONS: { value: View; label: string }[] = [
  { value: 'ensemble', label: '集成 (Ensemble)' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'alstm', label: 'ALSTM' },
  { value: 'tra', label: 'TRA' },
];

export function FilterBar({ params, resultCount, candidateCount, onChange, onReset }: FilterBarProps) {
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 space-y-5">
      {/* Header: result count + reset */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider">筛选</h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#8b949e]">
            {resultCount === null ? '加载中…' : `${resultCount} / ${candidateCount ?? '?'} 只`}
          </span>
          <button
            onClick={onReset}
            className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]"
          >
            重置
          </button>
        </div>
      </div>

      {/* Group 1: 基础 (immediate update) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">基础</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Select label="视图" value={params.view} options={VIEW_OPTIONS} onChange={(v) => onChange({ view: v as View })} />
          <NumberField label="Top N" value={params.top} min={1} max={300} onChange={(v) => onChange({ top: v })} />
          <NumberField label="窗口天数" value={params.days} min={1} max={60} onChange={(v) => onChange({ days: v })} />
          <NumberField label="最少进 top N 天数" value={params.min_top} min={0} max={params.days} onChange={(v) => onChange({ min_top: v })} />
        </div>
      </div>

      {/* Group 2: 价格 (debounced) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">价格 (¥/股)</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <DebouncedField label="最低单价" value={params.min_price} step={0.01} min={0} placeholder="无下限" onCommit={(v) => onChange({ min_price: v })} />
          <DebouncedField label="最高单价" value={params.max_price} step={0.01} min={0} placeholder="无上限" onCommit={(v) => onChange({ max_price: v })} />
        </div>
        <p className="text-[10px] text-[#6e7681] mt-1">
          A 股 100 股/手 · 4000 元买入 → 单价 ≤ ¥40 · ETF 可单股买
        </p>
      </div>

      {/* Group 3: 走势 (debounced) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">走势</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          <Select
            label="涨跌幅 N 日"
            value={String(params.pct_change_n)}
            options={PCT_CHANGE_N_OPTIONS.map((n) => ({ value: String(n), label: `${n} 日` }))}
            onChange={(v) => onChange({ pct_change_n: Number(v) as PctChangeN })}
          />
          <DebouncedField label="涨跌幅 min (%)" value={pctToUi(params.min_pct_change)} step={0.1} placeholder="不限" onCommit={(v) => onChange({ min_pct_change: pctFromUi(v) })} />
          <DebouncedField label="涨跌幅 max (%)" value={pctToUi(params.max_pct_change)} step={0.1} placeholder="不限" onCommit={(v) => onChange({ max_pct_change: pctFromUi(v) })} />
          <DebouncedField label="振幅 min (%)" value={pctToUi(params.min_amplitude)} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ min_amplitude: pctFromUi(v) })} />
          <DebouncedField label="振幅 max (%)" value={pctToUi(params.max_amplitude)} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ max_amplitude: pctFromUi(v) })} />
          <DebouncedField label="量比 min" value={params.min_vol_ratio} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ min_vol_ratio: v })} />
          <DebouncedField label="量比 max" value={params.max_vol_ratio} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ max_vol_ratio: v })} />
          <Select
            label="创 N 日新高"
            value={String(params.new_high_n)}
            options={NEW_HIGH_N_OPTIONS.map((n) => ({ value: String(n), label: n === 0 ? '关闭' : `${n} 日` }))}
            onChange={(v) => onChange({ new_high_n: Number(v) as NewHighN })}
          />
        </div>
      </div>

      {/* Group 4: 属性 (immediate) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">属性</h3>
        <div className="flex flex-col md:flex-row md:items-center gap-4">
          <BoardsCheckboxes value={params.boards} onChange={(boards) => onChange({ boards })} />
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={params.exclude_st}
              onChange={(e) => onChange({ exclude_st: e.target.checked })}
              className="accent-[#1f6feb]"
            />
            <span>排除 ST</span>
          </label>
        </div>
      </div>

      {/* Group 5: 共识 (immediate; UI-only filter, doesn't hit backend) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">共识</h3>
        <label className="block max-w-md">
          <span className="text-xs text-[#6e7681]">最低共识 ({params.min_consensus.toFixed(2)})</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={params.min_consensus}
            onChange={(e) => onChange({ min_consensus: Number(e.target.value) })}
            className="mt-1 w-full h-9 accent-[#1f6feb]"
          />
        </label>
      </div>
    </div>
  );
}

// === sub-components ===

function NumberField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      />
    </label>
  );
}

function Select<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </label>
  );
}

/** Numeric field that debounces user typing before committing to the parent.
 *  500ms debounce; commits empty string as null. */
function DebouncedField({
  label, value, step, min, placeholder, onCommit,
}: {
  label: string;
  value: number | null;
  step?: number;
  min?: number;
  placeholder?: string;
  onCommit: (v: number | null) => void;
}) {
  const [local, setLocal] = useState<string>(value === null ? '' : String(value));

  // External -> local: keep in sync if parent value changes via reset / URL
  useEffect(() => {
    setLocal(value === null ? '' : String(value));
  }, [value]);

  // Local -> external (debounced)
  useEffect(() => {
    const handle = setTimeout(() => {
      if (local === '') {
        if (value !== null) onCommit(null);
        return;
      }
      const parsed = Number(local);
      if (!Number.isFinite(parsed)) return;
      if (parsed !== value) onCommit(parsed);
    }, 500);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [local]);

  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <input
        type="number"
        value={local}
        step={step}
        min={min}
        placeholder={placeholder}
        onChange={(e) => setLocal(e.target.value)}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      />
    </label>
  );
}

function BoardsCheckboxes({ value, onChange }: { value: Board[]; onChange: (next: Board[]) => void }) {
  const toggle = (b: Board) => {
    const set = new Set(value);
    if (set.has(b)) set.delete(b); else set.add(b);
    onChange(Array.from(set));
  };
  return (
    <fieldset>
      <legend className="text-xs text-[#6e7681] uppercase tracking-wider mb-1">板块 (多选 = 并集)</legend>
      <div className="flex flex-wrap gap-3">
        {BOARDS.map(({ value: b, label }) => (
          <label key={b} className="flex items-center gap-1 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={value.includes(b)}
              onChange={() => toggle(b)}
              className="accent-[#1f6feb]"
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

// pct utility: backend uses raw decimal (0.05 = 5%), UI shows percent (5.0)
// e.g. pctToUi(0.0532) === 5.32, pctFromUi(5.32) === 0.0532.
function pctToUi(v: number | null): number | null {
  return v === null ? null : Math.round(v * 10000) / 100;
}

function pctFromUi(v: number | null): number | null {
  return v === null ? null : v / 100;
}
