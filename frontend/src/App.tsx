import { Route, Routes } from 'react-router-dom';
import Layout from '@/components/Layout';
import ChartPage from '@/charts/ChartPage';
import Dashboard from '@/pages/Dashboard';
import Evaluation from '@/pages/Evaluation';
import EvaluationCompare from '@/pages/EvaluationCompare';
import EvaluationDetail from '@/pages/EvaluationDetail';
import Picks from '@/pages/Picks';
import Portfolio from '@/pages/Portfolio';
import PortfolioTransactions from '@/pages/PortfolioTransactions';
import Settings from '@/pages/Settings';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/charts/:symbol" element={<ChartPage />} />
        <Route path="/picks" element={<Picks />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/portfolio/transactions" element={<PortfolioTransactions />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/evaluation/compare" element={<EvaluationCompare />} />
        <Route path="/evaluation/:recorderId" element={<EvaluationDetail />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
