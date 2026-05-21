import { Route, Routes } from 'react-router-dom';
import Layout from '@/components/Layout';
import ChartPage from '@/charts/ChartPage';
import Dashboard from '@/pages/Dashboard';
import Picks from '@/pages/Picks';
import Portfolio from '@/pages/Portfolio';
import PortfolioTransactions from '@/pages/PortfolioTransactions';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/charts/:symbol" element={<ChartPage />} />
        <Route path="/picks" element={<Picks />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/portfolio/transactions" element={<PortfolioTransactions />} />
      </Route>
    </Routes>
  );
}
