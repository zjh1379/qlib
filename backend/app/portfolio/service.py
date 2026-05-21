from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.core.qlib_adapter import (
    get_calendar_end,
    get_market_with_names,
    get_ohlcv,
    list_available_markets,
)
from app.portfolio.orm import TransactionORM
from app.portfolio.schemas import (
    Holding,
    HoldingsResponse,
    Transaction,
    TransactionIn,
    TransactionUpdate,
)


def _orm_to_schema(row: TransactionORM) -> Transaction:
    return Transaction.model_validate(row)


async def _aggregate_for_symbol(session: AsyncSession, symbol: str) -> dict:
    """Return aggregated buy/sell totals for a single symbol (used by validation)."""
    res = await session.execute(
        text(
            """
            SELECT
                COALESCE(SUM(CASE WHEN kind='buy'  THEN qty END), 0)               AS bought,
                COALESCE(SUM(CASE WHEN kind='sell' THEN qty END), 0)               AS sold,
                COALESCE(SUM(CASE WHEN kind='buy'  THEN qty*price + fee END), 0)   AS cost_in,
                COALESCE(SUM(CASE WHEN kind='sell' THEN qty*price - fee END), 0)   AS cash_out
            FROM transactions
            WHERE symbol = :symbol
            """
        ),
        {"symbol": symbol},
    )
    row = res.mappings().first() or {}
    return {
        "bought": float(row.get("bought") or 0),
        "sold": float(row.get("sold") or 0),
        "cost_in": float(row.get("cost_in") or 0),
        "cash_out": float(row.get("cash_out") or 0),
    }


def _validate_sell_does_not_oversell(
    agg: dict, new_sell_qty: float, *, exclude_qty: float = 0.0
) -> None:
    """Raise ConflictError if a sell of `new_sell_qty` would exceed available qty.

    `exclude_qty` is the qty being replaced (e.g., during update of an existing sell).
    """
    available = agg["bought"] - agg["sold"] + exclude_qty
    if new_sell_qty > available + 1e-9:
        raise ConflictError(
            f"sell qty {new_sell_qty} exceeds available {available}",
            code="oversell",
            context={"available": available, "requested": new_sell_qty},
        )


async def add_transaction(session: AsyncSession, payload: TransactionIn) -> Transaction:
    if payload.kind == "sell":
        agg = await _aggregate_for_symbol(session, payload.symbol)
        _validate_sell_does_not_oversell(agg, payload.qty)

    row = TransactionORM(
        symbol=payload.symbol,
        kind=payload.kind,
        qty=payload.qty,
        price=payload.price,
        fee=payload.fee,
        executed_at=payload.executed_at,
        broker=payload.broker,
        notes=payload.notes,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return _orm_to_schema(row)


async def list_transactions(
    session: AsyncSession,
    symbol: str | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> list[Transaction]:
    clauses = []
    params: dict = {}
    if symbol is not None:
        clauses.append("symbol = :symbol")
        params["symbol"] = symbol
    if from_date is not None:
        clauses.append("executed_at >= :from_date")
        params["from_date"] = from_date
    if to_date is not None:
        clauses.append("executed_at <= :to_date")
        params["to_date"] = to_date
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM transactions {where} ORDER BY executed_at ASC, id ASC"

    res = await session.execute(text(sql), params)
    rows = res.mappings().all()
    return [
        Transaction(
            id=r["id"],
            symbol=r["symbol"],
            kind=r["kind"],
            qty=r["qty"],
            price=r["price"],
            fee=r["fee"],
            executed_at=r["executed_at"],
            broker=r["broker"],
            notes=r["notes"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


async def get_transaction(session: AsyncSession, tx_id: int) -> Transaction:
    row = await session.get(TransactionORM, tx_id)
    if row is None:
        raise NotFoundError(
            f"transaction {tx_id} not found",
            code="transaction_missing",
            context={"id": tx_id},
        )
    return _orm_to_schema(row)


async def update_transaction(
    session: AsyncSession, tx_id: int, payload: TransactionUpdate
) -> Transaction:
    row = await session.get(TransactionORM, tx_id)
    if row is None:
        raise NotFoundError(
            f"transaction {tx_id} not found",
            code="transaction_missing",
            context={"id": tx_id},
        )

    # Determine new qty / kind to validate oversell when this is a sell-row
    new_qty = payload.qty if payload.qty is not None else row.qty
    if row.kind == "sell":
        agg = await _aggregate_for_symbol(session, row.symbol)
        _validate_sell_does_not_oversell(agg, new_qty, exclude_qty=row.qty)

    if payload.qty is not None:
        row.qty = payload.qty
    if payload.price is not None:
        row.price = payload.price
    if payload.fee is not None:
        row.fee = payload.fee
    if payload.executed_at is not None:
        row.executed_at = payload.executed_at
    if payload.broker is not None:
        row.broker = payload.broker
    if payload.notes is not None:
        row.notes = payload.notes

    await session.commit()
    await session.refresh(row)
    return _orm_to_schema(row)


async def delete_transaction(session: AsyncSession, tx_id: int) -> None:
    row = await session.get(TransactionORM, tx_id)
    if row is None:
        raise NotFoundError(
            f"transaction {tx_id} not found",
            code="transaction_missing",
            context={"id": tx_id},
        )
    await session.delete(row)
    await session.commit()


def _build_name_lookup() -> dict[str, str]:
    """Build a symbol -> Chinese name lookup over all known markets.

    Best-effort: returns an empty dict if anything fails (e.g. qlib not initialized).
    """
    name_map: dict[str, str] = {}
    try:
        for mi in list_available_markets():
            try:
                for item in get_market_with_names(mi["name"]):
                    sym = item.get("symbol")
                    nm = item.get("name") or ""
                    if sym and (sym not in name_map or not name_map[sym]):
                        name_map[sym] = nm
            except Exception:
                continue
    except Exception:
        pass
    return name_map


def _fetch_current_prices(symbols: list[str]) -> tuple[dict[str, float], str | None]:
    """Return ({symbol: close_price}, as_of_date_str). Best-effort — empty on failure."""
    if not symbols:
        return {}, None
    try:
        cal_end = get_calendar_end()
        as_of = str(cal_end)
        start = as_of
        df = get_ohlcv(symbols, start=start, end=as_of, freq="day")
        # df is a MultiIndex (instrument, datetime) DataFrame (or datetime, instrument).
        # We want the last close per symbol.
        prices: dict[str, float] = {}
        # Try the qlib convention: index is (instrument, datetime) with columns '$close', etc.
        try:
            # Use last row per instrument
            close = df["$close"]
            # close has a MultiIndex; group by instrument-level
            # qlib returns index (instrument, datetime)
            if hasattr(close.index, "names") and "instrument" in (close.index.names or []):
                grouped = close.groupby(level="instrument").last()
                for sym, val in grouped.items():
                    if val is not None:
                        prices[str(sym)] = float(val)
            else:
                # Fallback: iterate over all rows and pick last per symbol
                for idx, val in close.items():
                    if isinstance(idx, tuple) and len(idx) >= 1:
                        sym = str(idx[0])
                        prices[sym] = float(val)
        except Exception:
            return {}, as_of
        return prices, as_of
    except Exception:
        return {}, None


async def get_holdings(session: AsyncSession) -> HoldingsResponse:
    # 1) Aggregate per symbol
    res = await session.execute(
        text(
            """
            SELECT
                symbol,
                COALESCE(SUM(CASE WHEN kind='buy'  THEN qty END), 0)             AS bought,
                COALESCE(SUM(CASE WHEN kind='sell' THEN qty END), 0)             AS sold,
                COALESCE(SUM(CASE WHEN kind='buy'  THEN qty*price + fee END), 0) AS cost_in,
                COALESCE(SUM(CASE WHEN kind='sell' THEN qty*price - fee END), 0) AS cash_out
            FROM transactions
            GROUP BY symbol
            ORDER BY symbol ASC
            """
        )
    )
    aggregated = res.mappings().all()

    raw_holdings: list[dict] = []
    for r in aggregated:
        bought = float(r["bought"] or 0)
        sold = float(r["sold"] or 0)
        cost_in = float(r["cost_in"] or 0)
        cash_out = float(r["cash_out"] or 0)
        qty = bought - sold
        if qty <= 1e-9:
            continue
        avg_cost = cost_in / bought if bought > 0 else 0.0
        effective_cost = (cost_in - cash_out) / qty if qty > 0 else 0.0
        raw_holdings.append(
            {
                "symbol": r["symbol"],
                "qty": qty,
                "avg_cost": avg_cost,
                "effective_cost": effective_cost,
            }
        )

    symbols = [h["symbol"] for h in raw_holdings]

    # Name lookup (best-effort)
    name_map = _build_name_lookup()

    # Current prices (best-effort)
    prices, as_of = _fetch_current_prices(symbols)

    holdings: list[Holding] = []
    total_cost = 0.0
    total_mv: float | None = 0.0 if prices else None
    total_pnl: float | None = 0.0 if prices else None
    any_priced = False

    for h in raw_holdings:
        sym = h["symbol"]
        qty = h["qty"]
        eff = h["effective_cost"]
        cur = prices.get(sym)
        mv = qty * cur if cur is not None else None
        pnl = (mv - qty * eff) if mv is not None else None
        denom = qty * eff
        pnl_pct = (pnl / denom) if (pnl is not None and denom > 0) else None

        holdings.append(
            Holding(
                symbol=sym,
                name=name_map.get(sym, ""),
                qty=qty,
                avg_cost=h["avg_cost"],
                effective_cost=eff,
                current_price=cur,
                market_value=mv,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
            )
        )
        total_cost += qty * eff
        if mv is not None and total_mv is not None:
            total_mv += mv
            any_priced = True
        if pnl is not None and total_pnl is not None:
            total_pnl += pnl

    if not any_priced:
        total_mv = None
        total_pnl = None

    return HoldingsResponse(
        holdings=holdings,
        total_cost=total_cost,
        total_market_value=total_mv,
        total_unrealized_pnl=total_pnl,
        as_of=as_of,
    )
