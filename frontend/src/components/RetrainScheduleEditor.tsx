import { useState } from 'react';
import { ApiError } from '@/api/client';
import {
  type RetrainScheduleUpdate,
  useRetrainSchedule,
  useRunRetrainNow,
  useUpdateRetrainSchedule,
} from '@/scheduling/hooks';
import { cn } from '@/lib/utils';

const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function formatTs(ts: string | null | undefined): string {
  if (!ts) return '—';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    return `${err.detail}${err.code && err.code !== 'unknown' ? ` (${err.code})` : ''}`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

export default function RetrainScheduleEditor() {
  const { data, isPending, error } = useRetrainSchedule();
  const saveMut = useUpdateRetrainSchedule();
  const runNowMut = useRunRetrainNow();

  const [draft, setDraft] = useState<RetrainScheduleUpdate | null>(null);

  const effective: RetrainScheduleUpdate =
    draft ??
    (data
      ? {
          day_of_week: data.day_of_week,
          hour: data.hour,
          minute: data.minute,
          enabled: data.enabled,
        }
      : { day_of_week: 6, hour: 22, minute: 0, enabled: true });

  if (isPending) {
    return (
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 text-sm text-[#8b949e]">
        加载中…
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-400">
        加载排程失败: {errorMessage(error)}
      </div>
    );
  }

  const isDirty = draft !== null;

  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 space-y-4 max-w-3xl">
      <div>
        <h3 className="text-lg font-semibold">每周自动重训练</h3>
        <p className="text-xs text-[#8b949e] mt-1">
          交易时段（周一至周五 09:00–15:00 中国时区）期间禁止触发。
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <label className="block">
          <span className="text-xs text-[#6e7681] uppercase tracking-wider">星期</span>
          <select
            className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
            value={effective.day_of_week}
            onChange={(e) =>
              setDraft({ ...effective, day_of_week: Number(e.target.value) })
            }
          >
            {DOW.map((d, i) => (
              <option key={i} value={i}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="text-xs text-[#6e7681] uppercase tracking-wider">小时 (0–23)</span>
          <input
            type="number"
            min={0}
            max={23}
            value={effective.hour}
            onChange={(e) =>
              setDraft({
                ...effective,
                hour: Math.max(0, Math.min(23, Number(e.target.value) || 0)),
              })
            }
            className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
          />
        </label>
        <label className="block">
          <span className="text-xs text-[#6e7681] uppercase tracking-wider">分钟 (0–59)</span>
          <input
            type="number"
            min={0}
            max={59}
            value={effective.minute}
            onChange={(e) =>
              setDraft({
                ...effective,
                minute: Math.max(0, Math.min(59, Number(e.target.value) || 0)),
              })
            }
            className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
          />
        </label>
        <label className="flex items-end gap-2 pb-1">
          <input
            type="checkbox"
            checked={effective.enabled}
            onChange={(e) => setDraft({ ...effective, enabled: e.target.checked })}
            className="h-4 w-4 rounded border-[#30363d] bg-[#161b22] accent-[#1f6feb]"
          />
          <span className="text-sm">启用</span>
        </label>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <button
          type="button"
          disabled={!isDirty || saveMut.isPending}
          onClick={() =>
            saveMut.mutate(effective, {
              onSuccess: () => setDraft(null),
            })
          }
          className={cn(
            'px-4 py-2 rounded-md text-sm font-medium transition',
            !isDirty || saveMut.isPending
              ? 'bg-[#21262d] text-[#6e7681] cursor-not-allowed'
              : 'bg-[#1f6feb] text-white hover:bg-[#1f6febcc]',
          )}
        >
          {saveMut.isPending ? '保存中…' : '保存'}
        </button>
        {isDirty && (
          <button
            type="button"
            onClick={() => setDraft(null)}
            className="px-3 py-2 rounded-md text-sm bg-[#21262d] text-[#e6edf3] hover:bg-[#30363d]"
          >
            取消
          </button>
        )}
        <button
          type="button"
          disabled={runNowMut.isPending}
          onClick={() => runNowMut.mutate(false)}
          className={cn(
            'px-4 py-2 rounded-md text-sm font-medium transition',
            runNowMut.isPending
              ? 'bg-[#21262d] text-[#6e7681] cursor-not-allowed'
              : 'bg-[#bd561d] text-white hover:bg-[#bd561dcc]',
          )}
        >
          {runNowMut.isPending ? '触发中…' : '立即运行'}
        </button>
      </div>

      {saveMut.error && (
        <div className="rounded-md border border-red-900 bg-red-950/30 p-3 text-sm text-red-400">
          保存失败: {errorMessage(saveMut.error)}
        </div>
      )}

      {runNowMut.data?.status === 'rejected' && (
        <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-300 flex items-center gap-3 flex-wrap">
          <span>
            被拒绝: {runNowMut.data.reason ?? '未知原因'}。可以点击 “强制运行” 覆盖。
          </span>
          <button
            type="button"
            disabled={runNowMut.isPending}
            onClick={() => runNowMut.mutate(true)}
            className="px-3 py-1 rounded-md text-xs bg-red-800 text-white hover:bg-red-700"
          >
            强制运行
          </button>
        </div>
      )}

      {runNowMut.data?.status === 'started' && (
        <div className="rounded-md border border-green-900 bg-green-950/30 p-3 text-sm text-green-400">
          已启动 job:{' '}
          <span className="font-mono text-xs">{runNowMut.data.job_id ?? '?'}</span>
        </div>
      )}

      {runNowMut.error && (
        <div className="rounded-md border border-red-900 bg-red-950/30 p-3 text-sm text-red-400">
          触发失败: {errorMessage(runNowMut.error)}
        </div>
      )}

      <div className="text-xs text-[#6e7681] grid grid-cols-2 gap-4 pt-2 border-t border-[#21262d]">
        <div>
          <span className="uppercase tracking-wider">上次运行</span>
          <div className="font-mono text-[#8b949e] mt-1">{formatTs(data?.last_run_at)}</div>
        </div>
        <div>
          <span className="uppercase tracking-wider">下次运行</span>
          <div className="font-mono text-[#8b949e] mt-1">{formatTs(data?.next_run_at)}</div>
        </div>
      </div>
    </div>
  );
}
