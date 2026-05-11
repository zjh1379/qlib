import { Link, Outlet } from 'react-router-dom';
import { cn } from '@/lib/utils';

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-[#30363d] px-6 py-3 flex items-center gap-6">
        <span className="font-semibold">Qlib Companion</span>
        <nav className="flex gap-4 text-sm">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/charts/SH600519">Charts</NavLink>
        </nav>
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
