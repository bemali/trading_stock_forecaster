import torch
import numpy as np
#import timesfm

"""torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# Batched forecast with two dummy inputs. One is linear, the other is a sinusodal wave. Two different context lengths. This demonstrates the ability of the model to tackle different input context lengths at once.
input_1 = np.linspace(0, 1, 100) 
input_2 = np.sin(np.linspace(0, 20, 67))

point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        input_1, # time index
        input_2, # a sine wave with 67 time steps
    ],  # Two dummy inputs
)
print(point_forecast.shape)  # (2, 12)
print(quantile_forecast.shape)  # (2, 12, 10): mean, then 10th to 90th quantiles."""

from src.inputData import fetch_yahoo_history, shift_months
from datetime import date, timedelta
import yfinance as yf

ticker = ["TSLA", "AAPL"]
end_date = date.today()
start_date = end_date - timedelta(days=10)  # 1 year of data

data = fetch_yahoo_history(
    tickers=ticker,
    start_date=start_date,
    end_date=end_date,
    field="close",
)
print(data)

