import type { FilterParams } from './types';

interface FilterBarProps {
  params: FilterParams;
  resultCount: number | null;       // null = loading
  candidateCount: number | null;    // null = loading
  onChange: (patch: Partial<FilterParams>) => void;
  onReset: () => void;
}

export function FilterBar({ params, resultCount, candidateCount, onChange, onReset }: FilterBarProps) {
  // params / onChange will be wired up to real inputs in Task 9. Reference
  // them here so noUnusedParameters does not complain in the meantime.
  void params;
  void onChange;
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider">筛选</h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#8b949e]">
            {resultCount === null
              ? '加载中…'
              : `${resultCount} / ${candidateCount ?? '?'} 只`}
          </span>
          <button
            onClick={onReset}
            className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]"
          >
            重置
          </button>
        </div>
      </div>

      {/* Existing fields (placeholder — Task 9 adds the new groups below) */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {/* T8 leaves these inputs as-is from Picks.tsx; T9 swaps them out. */}
        <div className="text-xs text-[#6e7681]">视图 / Top N / 窗口 / minTop 在 Task 9 添加</div>
      </div>
    </div>
  );
}
