import { Route, Routes } from 'react-router-dom';
import Layout from '@/components/Layout';
import ChartPage from '@/charts/ChartPage';

function Home() {
  return (
    <div className="space-y-2">
      <h1 className="text-xl font-semibold">Dashboard</h1>
      <p className="text-[#8b949e] text-sm">
        Open a chart by URL: <code>/charts/SH600519</code> · <code>/charts/SZ000001</code>
      </p>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/charts/:symbol" element={<ChartPage />} />
      </Route>
    </Routes>
  );
}
