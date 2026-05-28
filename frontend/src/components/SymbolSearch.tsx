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
const FULL_SYMBOL_RE = /^(SH|SZ)\d{6}$/i;
const BARE_DIGITS_RE = /^\d{6}$/;

/**
 * Infer the SH/SZ prefix from a bare 6-digit code based on CN convention:
 *
 *   SH (上海):
 *     60xxxx   主板A
 *     68xxxx   科创板
 *     51-58xxxx 沪市 ETF / 基金
 *     90xxxx   B股
 *
 *   SZ (深圳):
 *     000xxx   主板A
 *     001xxx   主板A
 *     002xxx   中小板
 *     003xxx   主板
 *     30xxxx   创业板
 *     15-17xxxx 深市 ETF / 基金
 *     20xxxx   B股
 *
 * Returns null if the leading digits don't match a known segment.
 */
function inferPrefix(digits: string): 'SH' | 'SZ' | null {
  if (!BARE_DIGITS_RE.test(digits)) return null;
  const head2 = digits.slice(0, 2);
  const head3 = digits.slice(0, 3);
  // SH
  if (
    head2 === '60' ||
    head2 === '68' ||
    head2 === '51' ||
    head2 === '52' ||
    head2 === '53' ||
    head2 === '54' ||
    head2 === '55' ||
    head2 === '56' ||
    head2 === '57' ||
    head2 === '58' ||
    head2 === '90'
  ) {
    return 'SH';
  }
  // SZ
  if (
    head3 === '000' ||
    head3 === '001' ||
    head3 === '002' ||
    head3 === '003' ||
    head2 === '30' ||
    head2 === '15' ||
    head2 === '16' ||
    head2 === '17' ||
    head2 === '20'
  ) {
    return 'SZ';
  }
  return null;
}

/**
 * Normalize whatever the user typed into a candidate qlib symbol (e.g. SH600519).
 * Returns null if we can't form a valid one.
 */
function normalizeSymbol(input: string): string | null {
  const trimmed = input.trim().toUpperCase();
  if (FULL_SYMBOL_RE.test(trimmed)) return trimmed;
  if (BARE_DIGITS_RE.test(trimmed)) {
    const prefix = inferPrefix(trimmed);
    if (prefix) return prefix + trimmed;
  }
  return null;
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
  // Notify same-tab listeners (RecentlyViewed component) so they refresh
  // without needing a route change. `storage` event doesn't fire same-tab.
  try {
    window.dispatchEvent(new Event('qlib-recent-updated'));
  } catch {
    /* SSR / tests */
  }
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

  // Symbol normalized from user input (handles both SH600519 and bare 600519)
  const candidateSymbol = useMemo(() => normalizeSymbol(q), [q]);
  const noResultsButValidSymbol = filtered.length === 0 && candidateSymbol !== null;

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
          if (e.key === 'Enter') {
            if (filtered.length > 0) {
              go(filtered[0].symbol);
            } else if (candidateSymbol) {
              handleAdd(candidateSymbol);
            }
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
          {noResultsButValidSymbol && candidateSymbol && (
            <li
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => handleAdd(candidateSymbol)}
              className={cn(
                'cursor-pointer px-3 py-2 hover:bg-[#1f6feb] text-sm flex items-center justify-between gap-2',
                addMut.isPending && 'opacity-50 cursor-wait',
              )}
            >
              <span>
                📥 添加 <span className="font-mono">{candidateSymbol}</span>
                {candidateSymbol !== q.trim().toUpperCase() && (
                  <span className="text-xs text-[#6e7681] ml-1">
                    (推断自 {q.trim()})
                  </span>
                )}{' '}
                到下载列表
              </span>
              <span className="text-xs text-[#8b949e]">
                {addMut.isPending ? '抓取中…' : '点击下载'}
              </span>
            </li>
          )}
          {!noResultsButValidSymbol && (
            <li className="px-3 py-2 text-xs text-[#6e7681] border-t border-[#21262d]">
              提示：输入 6 位代码（如 <span className="font-mono">159995</span>）或完整代码（如 <span className="font-mono">SH600519</span>）。
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
