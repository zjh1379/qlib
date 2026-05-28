import { useTriggerInference } from '@/inference/hooks';

interface Props {
  staleDays: number;
  asOfDate: string;
  dataLatestDate: string;
}

export default function StalenessBanner({ staleDays, asOfDate, dataLatestDate }: Props) {
  const trigger = useTriggerInference();
  if (staleDays <= 0) return null;
  return (
    <div className="rounded-md border border-orange-800 bg-orange-950/40 px-4 py-2 text-sm text-orange-300 flex items-center justify-between gap-3 flex-wrap">
      <div>
        ⚠️ 数据已更新到 <span className="font-mono font-semibold text-orange-100">{dataLatestDate}</span>，
        但预测停留在 <span className="font-mono text-orange-100">{asOfDate}</span>
        （<span className="font-semibold">{staleDays}</span> 个交易日前）。
      </div>
      <button
        type="button"
        disabled={trigger.isPending}
        onClick={() => trigger.mutate(false)}
        className="px-3 py-1 rounded bg-orange-600 hover:bg-orange-500 text-white text-xs font-medium disabled:opacity-50"
      >
        {trigger.isPending ? '推理中…' : '立即重新推理 →'}
      </button>
    </div>
  );
}
