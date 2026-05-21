import { Link, Outlet } from 'react-router-dom';
import SymbolSearch from '@/components/SymbolSearch';
import { cn } from '@/lib/utils';

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-[#30363d] px-6 py-3 flex items-center gap-6">
        <Link to="/" className="font-semibold whitespace-nowrap">
          Qlib Companion
        </Link>
        <nav className="flex gap-4 text-sm flex-shrink-0">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/picks">选股</NavLink>
          <NavLink to="/portfolio">持仓</NavLink>
        </nav>
        <div className="flex-1 max-w-md ml-auto">
          <SymbolSearch size="sm" placeholder="搜索股票…" />
        </div>
      </header>
      <main className="flex-1 p-6">
        <Outlet />
      </main>
    </div>
  );
}

function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link to={to} className={cn('text-[#8b949e] hover:text-[#e6edf3] transition')}>
      {children}
    </Link>
  );
}
