from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    yf = None
    _YFINANCE_IMPORT_ERROR = exc
else:
    _YFINANCE_IMPORT_ERROR = None


FIELD_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}


def shift_months(input_date: date, months: int) -> date:
    """Return a date shifted by `months`, clamping the day to month-end when needed."""
    year = input_date.year + (input_date.month - 1 + months) // 12
    month = (input_date.month - 1 + months) % 12 + 1

    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    month_end_day = (next_month - date.resolution).day
    day = min(input_date.day, month_end_day)
    return date(year, month, day)


def fetch_yahoo_history(
    tickers: Iterable[str],
    start_date: date,
    end_date: date,
    field: str = "close",
) -> list[dict[str, str]]:
    """Fetch daily historical data from Yahoo Finance for each ticker in `tickers`."""
    if yf is None:  # pragma: no cover
        raise RuntimeError(
            "yfinance is not installed. Run `pip install yfinance` first."
        ) from _YFINANCE_IMPORT_ERROR

    ticker_list = list(tickers)
    output_key = field

    # yfinance treats `end` as exclusive, so add one day.
    data = yf.download(
        tickers=ticker_list,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        return []

    rows: dict[str, list] = {}
    rows["timeindex"] = data.index.strftime("%Y-%m-%d").tolist()

    if len(ticker_list) == 1:
        ticker = ticker_list[0]
        requested_column = (ticker,FIELD_MAP[field])
        if requested_column not in data.columns:
            return {}

        rows[ticker] = data[requested_column].tolist()

        return rows

    for ticker in ticker_list:
        if ticker not in data.columns.get_level_values(0).tolist():
            continue
        
        requested_column = (ticker,FIELD_MAP[field])
        if requested_column not in data.columns:
            continue

        rows[ticker] = data[requested_column].tolist()

    return rows


