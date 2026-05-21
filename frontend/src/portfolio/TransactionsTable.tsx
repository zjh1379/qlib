import { useState } from 'react';
import type { components } from '@/api/types.gen';
import { useDeleteTransaction } from '@/portfolio/hooks';
import { cn } from '@/lib/utils';

type Transaction = components['schemas']['Transaction'];

interface Props {
  transactions: Transaction[];
  emptyMessage?: string;
}

export default function TransactionsTable({
  transactions,
  emptyMessage = '暂无交易记录',
}: Props) {
  const delMut = useDeleteTransaction();
  const [confirmId, setConfirmId] = useState<number | null>(null);

  if (transactions.length === 0) {
    return (
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-8 text-center text-sm text-[#8b949e]">
        {emptyMessage}
      </div>
    );
  }

  const handleDelete = async (id: number) => {
    try {
      await delMut.mutateAsync(id);
      setConfirmId(null);
    } catch (e) {
      console.error('delete failed', e);
      setConfirmId(null);
    }
  };

  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-[#161b22] text-[#8b949e]">
          <tr>
            <Th>时间</Th>
            <Th>代码</Th>
            <Th>方向</Th>
            <Th align="right">数量</Th>
            <Th align="right">价格</Th>
            <Th align="right">费用</Th>
            <Th>备注</Th>
            <Th align="right">操作</Th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((t) => (
            <tr key={t.id} className="border-t border-[#21262d] hover:bg-[#161b22]">
              <Td>
                <span className="text-[#8b949e]">
                  {new Date(t.executed_at).toLocaleString('zh-CN')}
                </span>
              </Td>
              <Td mono>{t.symbol}</Td>
              <Td>
                <span
                  className={cn(
                    'px-2 py-0.5 rounded text-xs font-medium',
                    t.kind === 'buy'
                      ? 'bg-green-900/40 text-green-400'
                      : 'bg-red-900/40 text-red-400',
                  )}
                >
                  {t.kind === 'buy' ? '买入' : '卖出'}
                </span>
              </Td>
              <Td align="right" mono>
                {t.qty.toLocaleString('zh-CN')}
              </Td>
              <Td align="right" mono>
                {t.price.toFixed(3)}
              </Td>
              <Td align="right" mono>
                {t.fee.toFixed(2)}
              </Td>
              <Td className="max-w-xs truncate text-[#8b949e]">{t.notes || '—'}</Td>
              <Td align="right">
                {confirmId === t.id ? (
                  <span className="inline-flex gap-1">
                    <button
                      onClick={() => handleDelete(t.id)}
                      disabled={delMut.isPending}
                      className="px-2 py-0.5 rounded text-xs bg-red-600 text-white hover:bg-red-700 disabled:opacity-50"
                    >
                      确认
                    </button>
                    <button
                      onClick={() => setConfirmId(null)}
                      className="px-2 py-0.5 rounded text-xs bg-[#21262d] hover:bg-[#30363d]"
                    >
                      取消
                    </button>
                  </span>
                ) : (
                  <button
                    onClick={() => setConfirmId(t.id)}
                    className="text-xs text-[#8b949e] hover:text-red-400"
                    aria-label={`delete transaction ${t.id}`}
                  >
                    删除
                  </button>
                )}
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
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
