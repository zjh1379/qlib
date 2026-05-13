import { Route, Routes } from 'react-router-dom';
import Layout from '@/components/Layout';
import ChartPage from '@/charts/ChartPage';
import Dashboard from '@/pages/Dashboard';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/charts/:symbol" element={<ChartPage />} />
      </Route>
    </Routes>
  );
}
