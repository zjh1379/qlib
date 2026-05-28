import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

type ToastKind = 'info' | 'success' | 'warning' | 'error';

interface ToastEntry {
  id: number;
  kind: ToastKind;
  message: string;
  ttl: number;        // ms; -1 = sticky
  createdAt: number;
}

const STORE: { listeners: Array<(toasts: ToastEntry[]) => void>; toasts: ToastEntry[]; nextId: number } = {
  listeners: [],
  toasts: [],
  nextId: 1,
};

function notify() {
  for (const fn of STORE.listeners) fn(STORE.toasts);
}

export function toast(message: string, opts: { kind?: ToastKind; ttl?: number } = {}) {
  const kind: ToastKind = opts.kind ?? 'info';
  const ttl = opts.ttl ?? (kind === 'error' ? 8000 : kind === 'warning' ? 6000 : 4000);
  const entry: ToastEntry = {
    id: STORE.nextId++,
    kind,
    message,
    ttl,
    createdAt: Date.now(),
  };
  STORE.toasts = [...STORE.toasts, entry];
  notify();
  if (ttl > 0) {
    setTimeout(() => dismiss(entry.id), ttl);
  }
  return entry.id;
}

export function dismiss(id: number) {
  STORE.toasts = STORE.toasts.filter((t) => t.id !== id);
  notify();
}

toast.success = (msg: string, ttl?: number) => toast(msg, { kind: 'success', ttl });
toast.error = (msg: string, ttl?: number) => toast(msg, { kind: 'error', ttl });
toast.warning = (msg: string, ttl?: number) => toast(msg, { kind: 'warning', ttl });
toast.info = (msg: string, ttl?: number) => toast(msg, { kind: 'info', ttl });

/**
 * Mount once in Layout. Renders the toast stack in the bottom-right.
 * Survives route changes because it's outside the <Outlet />.
 */
export function Toaster() {
  const [toasts, setToasts] = useState<ToastEntry[]>(STORE.toasts);

  useEffect(() => {
    const fn = (next: ToastEntry[]) => setToasts(next);
    STORE.listeners.push(fn);
    return () => {
      STORE.listeners = STORE.listeners.filter((x) => x !== fn);
    };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={cn(
            'pointer-events-auto rounded-md border px-4 py-2 text-sm shadow-lg max-w-md',
            'transition-all animate-slide-in',
            t.kind === 'error' && 'border-red-700 bg-red-950 text-red-200',
            t.kind === 'warning' && 'border-yellow-700 bg-yellow-950 text-yellow-200',
            t.kind === 'success' && 'border-green-700 bg-green-950 text-green-200',
            t.kind === 'info' && 'border-[#30363d] bg-[#161b22] text-[#e6edf3]',
          )}
        >
          <div className="flex items-start gap-3">
            <span className="flex-1 whitespace-pre-wrap break-words">{t.message}</span>
            <button
              onClick={() => dismiss(t.id)}
              className="text-[#6e7681] hover:text-[#e6edf3] text-xs flex-shrink-0"
              aria-label="关闭"
            >
              ✕
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
