# Qlib Companion · Frontend

React 18 + Vite + Tailwind + shadcn-style components + TanStack Query + Lightweight Charts.

## Setup

```bash
npm install
```

## Dev

```bash
npm run dev          # http://localhost:5173 (proxies /api to :8000)
npm run gen:api      # regenerate types from backend /openapi.json
npm test
npm run typecheck
```

## Build

```bash
npm run build        # outputs frontend/dist/, served by backend
```

## Routes

| Route | Page |
| --- | --- |
| `/` | Dashboard (P1: stub) |
| `/charts/:symbol` | Single-ticker chart with prediction overlay |

P2-P4 add `/picks`, `/portfolio`, `/ops`.
