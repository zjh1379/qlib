import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTransactions } from '@/portfolio/hooks';
import TransactionForm from '@/portfolio/TransactionForm';
import TransactionsTable from '@/portfolio/TransactionsTable';
import { cn } from '@/lib/utils';

export default function PortfolioTransactions() {
  const [symbol, setSymbol] = useState('');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [showForm, setShowForm] = useState(false);

  const params: { symbol?: string; from?: string; to?: string } = {};
  if (symbol.trim()) params.symbol = symbol.trim().toUpperCase();
  if (from) params.from = from;
  if (to) params.to = to;

  const { data, isPending, error } = useTransactions(params);

  return (
    <div className="space-y-6 max-w-6xl">
      <header className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold">交易历史</h1>
          <p className="text-sm text-[#8b949e] mt-1">
            <Link to="/portfolio" className="text-[#58a6ff] hover:underline">
              ← 返回持仓
            </Link>
          </p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className={cn(
            'px-4 py-2 rounded-md text-sm font-medium',
            'bg-[#1f6feb] text-white hover:bg-[#1f6febcc]',
          )}
        >
          + 添加交易
        </button>
      </header>

      {/* Filters */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Field label="代码">
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="SH600519"
              className={inputCls}
              aria-label="filter symbol"
            />
          </Field>
          <Field label="起始日期">
            <input
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              className={inputCls}
              aria-label="filter from"
            />
          </Field>
          <Field label="结束日期">
            <input
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              className={inputCls}
              aria-label="filter to"
            />
          </Field>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-400">
          加载交易失败: {(error as Error).message}
        </div>
      ) : isPending ? (
        <div className="text-sm text-[#8b949e]">加载中…</div>
      ) : data ? (
        <>
          <div className="text-xs text-[#6e7681]">共 {data.length} 条记录</div>
          <TransactionsTable transactions={data} emptyMessage="无符合条件的交易记录" />
        </>
      ) : null}

      <TransactionForm open={showForm} onClose={() => setShowForm(false)} />
    </div>
  );
}

const inputCls =
  'w-full rounded-md bg-[#161b22] border border-[#30363d] text-[#e6edf3] placeholder-[#6e7681] h-9 px-3 text-sm focus:outline-none focus:border-[#1f6feb]';

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs text-[#8b949e] uppercase tracking-wider mb-1 block">
        {label}
      </span>
      {children}
    </label>
  );
}
