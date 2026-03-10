from datetime import date
from pathlib import Path

import numpy as np
import src.timesfm as timesfm
import torch

from src.inputData import fetch_yahoo_history, shift_months

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required to save forecast plots. "
        "Install it with `pip install matplotlib`."
    ) from exc


TICKERS = ["TSLA", "AAPL"]
OUTPUT_DIR = Path("outputs")
LOOKBACK_POINTS = 100
CONTEXT_POINTS = 95
FORECAST_HORIZON = LOOKBACK_POINTS - CONTEXT_POINTS


def load_series(tickers: list[str]) -> dict[str, dict[str, np.ndarray]]:
    end_date = date.today()
    start_date = shift_months(end_date, -8)
    raw_history = fetch_yahoo_history(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        field="close",
    )

    if not raw_history or "timeindex" not in raw_history:
        raise RuntimeError("No Yahoo Finance history returned for the requested tickers.")

    timeindex = np.asarray(raw_history["timeindex"])
    series_map: dict[str, dict[str, np.ndarray]] = {}

    for ticker in tickers:
        if ticker not in raw_history:
            raise RuntimeError(f"Missing close prices for {ticker}.")

        values = np.asarray(raw_history[ticker], dtype=np.float32)
        valid_mask = np.isfinite(values)
        valid_dates = timeindex[valid_mask]
        valid_values = values[valid_mask]

        if valid_values.size < LOOKBACK_POINTS:
            raise RuntimeError(
                f"{ticker} only returned {valid_values.size} valid points; "
                f"{LOOKBACK_POINTS} are required."
            )

        series_map[ticker] = {
            "dates": valid_dates[-LOOKBACK_POINTS:],
            "values": valid_values[-LOOKBACK_POINTS:],
        }

    return series_map


def build_model():
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=FORECAST_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def save_forecast_plot(
    ticker: str,
    dates: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
) -> None:
    train_dates = dates[:CONTEXT_POINTS]
    forecast_dates = dates[CONTEXT_POINTS:]
    train_values = actual[:CONTEXT_POINTS]
    actual_future = actual[CONTEXT_POINTS:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_dates, train_values, label="Context (first 50)", linewidth=2)
    ax.plot(forecast_dates, actual_future, label="Actual", linewidth=2)
    ax.plot(forecast_dates, forecast, label="Forecast", linestyle="--", linewidth=2)
    ax.set_title(f"{ticker} 50-step Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{ticker.lower()}_forecast_vs_actual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    series_map = load_series(TICKERS)
    model = build_model()

    contexts = [series_map[ticker]["values"][:CONTEXT_POINTS] for ticker in TICKERS]
    point_forecast, _ = model.forecast(horizon=FORECAST_HORIZON, inputs=contexts)

    for idx, ticker in enumerate(TICKERS):
        dates = series_map[ticker]["dates"]
        actual = series_map[ticker]["values"]
        forecast = point_forecast[idx][:FORECAST_HORIZON]
        save_forecast_plot(ticker=ticker, dates=dates, actual=actual, forecast=forecast)
        print(f"Saved {ticker} forecast plot to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
