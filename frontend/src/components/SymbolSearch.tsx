import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAddSymbol, useInstruments } from '@/data/hooks';
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
const SYMBOL_RE = /^(SH|SZ)\d{6}$/i;

function isValidSymbolFormat(s: string): boolean {
  return SYMBOL_RE.test(s.trim());
}

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
  const { data, isPending } = useInstruments('all');
  const items: Instrument[] = data?.items ?? [];

  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const addMut = useAddSymbol();

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

  const trimmedUpper = q.trim().toUpperCase();
  const noResultsButValidSymbol =
    filtered.length === 0 && isValidSymbolFormat(trimmedUpper);

  const go = (symbol: string) => {
    saveRecent(symbol);
    setQ('');
    setOpen(false);
    navigate(`/charts/${symbol}`);
  };

  const handleAdd = async (sym: string) => {
    try {
      await addMut.mutateAsync(sym);
      saveRecent(sym);
      setQ('');
      setOpen(false);
      navigate(`/charts/${sym}`);
    } catch (e) {
      // surfaced via addMut.isError below
      console.error('add symbol failed', e);
    }
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
      {open && filtered.length === 0 && q.trim().length > 0 && (
        <ul className="absolute z-50 mt-1 w-full rounded-md border border-[#30363d] bg-[#0d1117] shadow-xl">
          <li className="px-3 py-2 text-xs text-[#6e7681]">
            未找到与 "{q}" 匹配的股票
          </li>
          {noResultsButValidSymbol && (
            <li
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => handleAdd(trimmedUpper)}
              className={cn(
                'cursor-pointer px-3 py-2 hover:bg-[#1f6feb] text-sm flex items-center justify-between gap-2',
                addMut.isPending && 'opacity-50 cursor-wait',
              )}
            >
              <span>
                📥 添加 <span className="font-mono">{trimmedUpper}</span> 到下载列表
              </span>
              <span className="text-xs text-[#8b949e]">
                {addMut.isPending ? '抓取中…' : '点击下载'}
              </span>
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
