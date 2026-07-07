"""Synthetic daily sales data generator.

Produces a realistic retail/SaaS-style time series with:
  - an upward linear trend
  - weekly seasonality (weekend effect)
  - yearly seasonality (annual cycle)
  - random promo days that temporarily lift sales
  - a handful of fixed "holiday" spikes
  - Gaussian noise

This lets the whole pipeline run end-to-end with zero external
downloads. Swap in a real CSV any time — see README "Using a real
dataset" for the expected schema.
"""

import numpy as np
import pandas as pd


def generate_sales_data(config: dict, seed: int = 42) -> pd.DataFrame:
    cfg = config["data"]
    rng = np.random.default_rng(seed)

    n_days = cfg["n_days"]
    dates = pd.date_range(cfg["start_date"], periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = cfg["base_level"] + cfg["trend_per_day"] * t
    weekly = cfg["weekly_amplitude"] * np.sin(2 * np.pi * t / 7)
    yearly = cfg["yearly_amplitude"] * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, cfg["noise_std"], n_days)

    is_promo = rng.random(n_days) < cfg["promo_rate"]
    promo_effect = is_promo * cfg["promo_boost"]

    # A handful of fixed calendar "holiday" spikes each year (e.g. Black Friday-ish)
    is_holiday = np.zeros(n_days, dtype=bool)
    for year_start in range(0, n_days, 365):
        for offset in (45, 150, 330):  # arbitrary recurring spike days
            idx = year_start + offset
            if idx < n_days:
                is_holiday[idx] = True
    holiday_effect = is_holiday * cfg["holiday_boost"]

    sales = trend + weekly + yearly + noise + promo_effect + holiday_effect
    sales = np.clip(sales, a_min=0, a_max=None)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "is_promo": is_promo.astype(int),
        "is_holiday": is_holiday.astype(int),
    })
    return df


if __name__ == "__main__":
    from src.utils import load_config, ensure_dirs

    config = load_config()
    ensure_dirs(config)
    data = generate_sales_data(config)
    out_path = config["paths"]["data"]
    data.to_csv(out_path, index=False)
    print(f"Wrote {len(data)} rows to {out_path}")
