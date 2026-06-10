import RetrainScheduleEditor from './settings/RetrainScheduleEditor';

export default function Settings() {
  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold">设置</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          系统级排程与运行参数配置。
        </p>
      </header>

      <RetrainScheduleEditor />
    </div>
  );
}
