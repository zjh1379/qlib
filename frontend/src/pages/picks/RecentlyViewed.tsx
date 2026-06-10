import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { loadRecent } from '@/components/SymbolSearch';
import { useInstruments } from '@/data/hooks';
import { cn } from '@/lib/utils';

interface Props {
  className?: string;
  maxCount?: number;
}

/** Compact "最近查看" pills strip. Reads from the same localStorage key
 *  populated by `SymbolSearch.saveRecent()` + `ChartPage` on mount. */
export default function RecentlyViewed({ className, maxCount = 8 }: Props) {
  const [recent, setRecent] = useState<string[]>(() => loadRecent());
  const { data } = useInstruments('all');

  // Re-read whenever localStorage may have changed (e.g. coming back from
  // a chart view). The 'storage' event only fires for other tabs; for
  // same-tab changes we listen to a custom event dispatched by saveRecent.
  useEffect(() => {
    const refresh = () => setRecent(loadRecent());
    window.addEventListener('qlib-recent-updated', refresh);
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener('qlib-recent-updated', refresh);
      window.removeEventListener('storage', refresh);
    };
  }, []);

  if (recent.length === 0) return null;

  const list = recent.slice(0, maxCount);
  const nameOf = (sym: string) => {
    const hit = data?.items?.find((it) => it.symbol === sym);
    return hit?.name ?? '';
  };

  const clearAll = () => {
    localStorage.removeItem('qlib-recent-symbols');
    window.dispatchEvent(new Event('qlib-recent-updated'));
  };

  return (
    <div className={cn('rounded-lg border border-[#21262d] bg-[#0d1117]/60 p-3', className)}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-[#6e7681] uppercase tracking-wider">
          最近查看 ({list.length})
        </span>
        <button
          onClick={clearAll}
          className="text-[10px] text-[#6e7681] hover:text-[#e6edf3] transition-colors"
          title="清空浏览历史"
        >
          清空
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {list.map((sym) => {
          const name = nameOf(sym);
          return (
            <Link
              key={sym}
              to={`/charts/${sym}`}
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#161b22] border border-[#30363d] hover:border-[#1f6feb] hover:bg-[#1f2937] transition-colors text-xs"
              title={name ? `${sym} · ${name}` : sym}
            >
              <span className="font-mono text-[#58a6ff]">{sym}</span>
              {name && <span className="text-[#8b949e]">{name}</span>}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
