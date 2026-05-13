import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useInstruments } from '@/data/hooks';
import { cn } from '@/lib/utils';

interface Instrument {
  symbol: string;
  name: string;
}

interface Props {
  className?: string;
  placeholder?: string;
  size?: 'sm' | 'md';
}

const RECENT_KEY = 'qlib-recent-symbols';

export function loadRecent(): string[] {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
  } catch {
    return [];
  }
}

export function saveRecent(symbol: string) {
  const list = loadRecent().filter((s) => s !== symbol);
  list.unshift(symbol);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 10)));
}

export default function SymbolSearch({
  className,
  placeholder = '搜索股票（代码 / 名称）',
  size = 'md',
}: Props) {
  const { data, isPending } = useInstruments();
  const items: Instrument[] = data?.items ?? [];

  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  const filtered = useMemo(() => {
    const query = q.trim().toLowerCase();
    if (!query) return items.slice(0, 20);
    return items
      .filter(
        (it) =>
          it.symbol.toLowerCase().includes(query) ||
          it.name.toLowerCase().includes(query),
      )
      .slice(0, 20);
  }, [q, items]);

  const go = (symbol: string) => {
    saveRecent(symbol);
    setQ('');
    setOpen(false);
    navigate(`/charts/${symbol}`);
  };

  return (
    <div className={cn('relative', className)}>
      <input
        type="text"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && filtered.length > 0) {
            go(filtered[0].symbol);
          } else if (e.key === 'Escape') {
            setOpen(false);
          }
        }}
        placeholder={isPending ? '加载中…' : placeholder}
        className={cn(
          'w-full rounded-md bg-[#161b22] border border-[#30363d] text-[#e6edf3] placeholder-[#6e7681]',
          size === 'sm' ? 'h-8 px-3 text-sm' : 'h-10 px-3 text-base',
          'focus:outline-none focus:border-[#1f6feb]',
        )}
        aria-label="symbol search"
      />
      {open && filtered.length > 0 && (
        <ul className="absolute z-50 mt-1 max-h-80 w-full overflow-y-auto rounded-md border border-[#30363d] bg-[#0d1117] shadow-xl">
          {filtered.map((it) => (
            <li
              key={it.symbol}
              onMouseDown={(e) => e.preventDefault() /* keep focus */}
              onClick={() => go(it.symbol)}
              className="cursor-pointer px-3 py-2 hover:bg-[#1f6feb] text-sm flex justify-between gap-3"
            >
              <span className="font-mono">{it.symbol}</span>
              <span className="text-[#8b949e]">{it.name || '—'}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
