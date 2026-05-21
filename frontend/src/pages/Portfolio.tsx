import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useHoldings, useTransactions } from '@/portfolio/hooks';
import HoldingsTable from '@/portfolio/HoldingsTable';
import TransactionForm from '@/portfolio/TransactionForm';
import TransactionsTable from '@/portfolio/TransactionsTable';
import { cn } from '@/lib/utils';

export default function Portfolio() {
  const { data: holdings, isPending, error } = useHoldings();
  const { data: recentTxs } = useTransactions();
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="space-y-6 max-w-6xl">
      <header className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-semibold">持仓</h1>
          <p className="text-sm text-[#8b949e] mt-1">
            实时市值与浮动盈亏（按最新收盘价计算）
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/portfolio/transactions"
            className="px-3 py-2 rounded-md text-sm bg-[#21262d] hover:bg-[#30363d]"
          >
            查看交易历史
          </Link>
          <button
            onClick={() => setShowForm(true)}
            className={cn(
              'px-4 py-2 rounded-md text-sm font-medium',
              'bg-[#1f6feb] text-white hover:bg-[#1f6febcc]',
            )}
          >
            + 添加交易
          </button>
        </div>
      </header>

      {error ? (
        <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-400">
          加载持仓失败: {(error as Error).message}
        </div>
      ) : isPending ? (
        <div className="text-sm text-[#8b949e]">加载中…</div>
      ) : holdings ? (
        <HoldingsTable data={holdings} />
      ) : null}

      {/* Recent transactions */}
      {recentTxs && recentTxs.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider">
              最近交易
            </h2>
            <Link
              to="/portfolio/transactions"
              className="text-xs text-[#58a6ff] hover:underline"
            >
              查看全部 →
            </Link>
          </div>
          <TransactionsTable transactions={recentTxs.slice(0, 10)} />
        </div>
      )}

      <TransactionForm open={showForm} onClose={() => setShowForm(false)} />
    </div>
  );
}
