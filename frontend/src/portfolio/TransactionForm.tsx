import { useState } from 'react';
import { useAddTransaction } from '@/portfolio/hooks';
import { cn } from '@/lib/utils';

interface Props {
  open: boolean;
  onClose: () => void;
  initialSymbol?: string;
}

const SYMBOL_RE = /^(SH|SZ)\d{6}$/i;

function nowLocalForInput(): string {
  const d = new Date();
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export default function TransactionForm({ open, onClose, initialSymbol = '' }: Props) {
  const addMut = useAddTransaction();
  const [symbol, setSymbol] = useState(initialSymbol);
  const [kind, setKind] = useState<'buy' | 'sell'>('buy');
  const [qty, setQty] = useState('');
  const [price, setPrice] = useState('');
  const [fee, setFee] = useState('0');
  const [executedAt, setExecutedAt] = useState(nowLocalForInput());
  const [broker, setBroker] = useState('');
  const [notes, setNotes] = useState('');
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const reset = () => {
    setSymbol(initialSymbol);
    setKind('buy');
    setQty('');
    setPrice('');
    setFee('0');
    setExecutedAt(nowLocalForInput());
    setBroker('');
    setNotes('');
    setError(null);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const validate = (): { ok: boolean; msg?: string } => {
    const sym = symbol.trim().toUpperCase();
    if (!sym) return { ok: false, msg: '代码不能为空' };
    if (!SYMBOL_RE.test(sym)) return { ok: false, msg: '代码格式无效，需如 SH600519' };
    const qtyN = Number(qty);
    if (!qty || !Number.isFinite(qtyN) || qtyN <= 0)
      return { ok: false, msg: '数量必须大于 0' };
    const priceN = Number(price);
    if (!price || !Number.isFinite(priceN) || priceN <= 0)
      return { ok: false, msg: '价格必须大于 0' };
    const feeN = Number(fee);
    if (!Number.isFinite(feeN) || feeN < 0) return { ok: false, msg: '费用不能为负' };
    if (!executedAt) return { ok: false, msg: '请选择执行时间' };
    return { ok: true };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const v = validate();
    if (!v.ok) {
      setError(v.msg ?? '校验失败');
      return;
    }
    try {
      // executedAt comes from datetime-local; convert to ISO string with seconds.
      const isoExecutedAt = new Date(executedAt).toISOString();
      await addMut.mutateAsync({
        symbol: symbol.trim().toUpperCase(),
        kind,
        qty: Number(qty),
        price: Number(price),
        fee: Number(fee),
        executed_at: isoExecutedAt,
        broker: broker.trim() || null,
        notes: notes.trim() || null,
      });
      reset();
      onClose();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-start justify-center p-4 overflow-y-auto"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) handleClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label="add transaction"
    >
      <div className="w-full max-w-lg mt-12 rounded-lg border border-[#30363d] bg-[#0d1117] shadow-xl">
        <div className="border-b border-[#30363d] px-5 py-3 flex items-center justify-between">
          <h2 className="text-base font-semibold">添加交易</h2>
          <button
            type="button"
            onClick={handleClose}
            className="text-[#8b949e] hover:text-[#e6edf3] text-lg leading-none"
            aria-label="close"
          >
            ×
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <Field label="代码" required>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="SH600519"
              className={inputCls}
              autoFocus
              aria-label="symbol"
            />
          </Field>

          <Field label="方向" required>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="kind"
                  value="buy"
                  checked={kind === 'buy'}
                  onChange={() => setKind('buy')}
                />
                <span className="text-green-400">买入</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="kind"
                  value="sell"
                  checked={kind === 'sell'}
                  onChange={() => setKind('sell')}
                />
                <span className="text-red-400">卖出</span>
              </label>
            </div>
          </Field>

          <div className="grid grid-cols-2 gap-3">
            <Field label="数量" required>
              <input
                type="number"
                value={qty}
                onChange={(e) => setQty(e.target.value)}
                placeholder="100"
                step="any"
                min="0"
                className={inputCls}
                aria-label="qty"
              />
            </Field>
            <Field label="价格" required>
              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="1500.00"
                step="any"
                min="0"
                className={inputCls}
                aria-label="price"
              />
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Field label="费用">
              <input
                type="number"
                value={fee}
                onChange={(e) => setFee(e.target.value)}
                step="any"
                min="0"
                className={inputCls}
                aria-label="fee"
              />
            </Field>
            <Field label="执行时间" required>
              <input
                type="datetime-local"
                value={executedAt}
                onChange={(e) => setExecutedAt(e.target.value)}
                className={inputCls}
                aria-label="executed_at"
              />
            </Field>
          </div>

          <Field label="券商 (可选)">
            <input
              type="text"
              value={broker}
              onChange={(e) => setBroker(e.target.value)}
              className={inputCls}
              aria-label="broker"
            />
          </Field>

          <Field label="备注 (可选)">
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              className={cn(inputCls, 'resize-none')}
              aria-label="notes"
            />
          </Field>

          {error && (
            <div className="text-sm text-red-400" role="alert">
              {error}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 rounded-md text-sm bg-[#21262d] hover:bg-[#30363d]"
            >
              取消
            </button>
            <button
              type="submit"
              disabled={addMut.isPending}
              className={cn(
                'px-4 py-2 rounded-md text-sm font-medium',
                'bg-[#1f6feb] text-white hover:bg-[#1f6febcc]',
                'disabled:bg-[#30363d] disabled:text-[#6e7681] disabled:cursor-not-allowed',
              )}
            >
              {addMut.isPending ? '提交中…' : '提交'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

const inputCls =
  'w-full rounded-md bg-[#161b22] border border-[#30363d] text-[#e6edf3] placeholder-[#6e7681] h-9 px-3 text-sm focus:outline-none focus:border-[#1f6feb]';

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs text-[#8b949e] uppercase tracking-wider mb-1 block">
        {label}
        {required && <span className="text-red-400 ml-1">*</span>}
      </span>
      {children}
    </label>
  );
}
