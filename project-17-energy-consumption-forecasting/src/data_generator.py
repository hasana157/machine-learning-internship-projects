"""
Synthetic daily energy consumption generator.

Simulates a realistic household/building load profile made of:
  - a slow long-term trend (e.g. growing appliance load)
  - weekly seasonality (weekday vs weekend usage)
  - yearly seasonality (summer/winter heating & cooling swings)
  - gaussian measurement/behavioural noise

Swap this out for a real dataset at any time -- see README "Using a
Real Dataset" section. The rest of the pipeline only cares about a
CSV with `date` and `consumption` columns.
"""

import numpy as np
import pandas as pd


def generate_energy_data(config: dict) -> pd.DataFrame:
    cfg = config["data"]
    n_days = cfg["n_days"]
    rng = np.random.default_rng(cfg["random_state"])

    dates = pd.date_range(cfg["start_date"], periods=n_days, freq="D")

    trend = np.linspace(0, cfg["trend_amplitude"], n_days)
    weekly = cfg["weekly_amplitude"] * np.sin(2 * np.pi * np.arange(n_days) / 7)
    yearly = cfg["yearly_amplitude"] * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = rng.normal(0, cfg["noise_sigma"], n_days)

    consumption = cfg["base_load"] + trend + weekly + yearly + noise

    return pd.DataFrame({"date": dates, "consumption": consumption})


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config = yaml.safe_load(open("config.yaml"))
    df = generate_energy_data(config)

    out_path = Path(config["paths"]["data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
