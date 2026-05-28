import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from '@/App';
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { toast } from '@/jobs/toast';

// Global mutation error handler: any useMutation whose component
// unmounts before the request resolves would silently lose its error.
// Surfacing them via the global toast queue means even if you've
// navigated away the failure reaches you.
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, retry: 1 } },
  mutationCache: new MutationCache({
    onError: (error, _vars, _ctx, mutation) => {
      // Skip if the mutation already has a local onError handler that
      // sets `meta.suppressGlobalToast: true`.
      if (mutation.meta?.suppressGlobalToast) return;
      const msg = (error as Error)?.message ?? '操作失败';
      toast.error(msg);
    },
  }),
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
