"""
Data generation module for SentinelFlow.
Generates multi-sensor time-series data with synthetic anomalies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

from src.utils import setup_logger

logger = setup_logger(__name__)

def inject_anomalies(df: pd.DataFrame, rng: np.random.Generator, config: dict) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, str]]:
    """
    Inject synthetic anomalies into the generated sensor data.

    Args:
        df (pd.DataFrame): Normal sensor data.
        rng (np.random.Generator): Random number generator instance.
        config (dict): Configuration dictionary.

    Returns:
        Tuple[pd.DataFrame, Dict[pd.Timestamp, str]]: DataFrame with anomalies and dictionary of anomaly labels.
    """
    df = df.copy()
    n_points = len(df)
    anomaly_rate = config["data"]["anomaly_rate"]
    n_anomalies = int(n_points * anomaly_rate)
    
    df["is_anomaly"] = 0
    anomaly_labels = {}
    
    # Select random indices for anomalies (avoiding edges to allow patterns)
    anomaly_indices = rng.choice(range(10, n_points - 10), size=n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        ts = df.index[idx]
        anomaly_type_rand = rng.random()
        
        # Temperature Anomalies
        if anomaly_type_rand < 0.25:
            if rng.random() > 0.5:
                # Sudden spike
                df.iloc[idx, df.columns.get_loc("temp")] += 25.0
                anomaly_labels[ts] = "Temp Spike"
            else:
                # Gradual drift (+0.5C per hour for 6h = 24 points at 15m intervals)
                drift_len = min(24, n_points - idx)
                drift_vals = np.linspace(0, 0.5 * 6, drift_len)
                df.iloc[idx:idx+drift_len, df.columns.get_loc("temp")] += drift_vals
                df.iloc[idx:idx+drift_len, df.columns.get_loc("is_anomaly")] = 1
                anomaly_labels[ts] = "Temp Drift Start"
                
        # Vibration Anomalies
        elif anomaly_type_rand < 0.50:
            if rng.random() > 0.5:
                # High-frequency burst (10x normal for 3-5 points)
                burst_len = rng.integers(3, 6)
                burst_len = min(burst_len, n_points - idx)
                df.iloc[idx:idx+burst_len, df.columns.get_loc("vibration")] *= 10
                df.iloc[idx:idx+burst_len, df.columns.get_loc("is_anomaly")] = 1
                anomaly_labels[ts] = "Vibration Burst"
            else:
                # Flat-line (stuck sensor for 5-10 points)
                stuck_len = rng.integers(5, 11)
                stuck_len = min(stuck_len, n_points - idx)
                stuck_val = df.iloc[idx-1]["vibration"]
                df.iloc[idx:idx+stuck_len, df.columns.get_loc("vibration")] = stuck_val
                df.iloc[idx:idx+stuck_len, df.columns.get_loc("is_anomaly")] = 1
                anomaly_labels[ts] = "Vibration Flatline"
                
        # Pressure Anomalies
        elif anomaly_type_rand < 0.75:
            if rng.random() > 0.5:
                # Pressure drop (suddenly falls to ~60 PSI)
                df.iloc[idx, df.columns.get_loc("pressure")] = 60.0 + rng.normal(0, 2)
                anomaly_labels[ts] = "Pressure Drop"
            else:
                # Slow leak (linear decay over 12 points)
                leak_len = min(12, n_points - idx)
                leak_vals = np.linspace(0, -20, leak_len)
                df.iloc[idx:idx+leak_len, df.columns.get_loc("pressure")] += leak_vals
                df.iloc[idx:idx+leak_len, df.columns.get_loc("is_anomaly")] = 1
                anomaly_labels[ts] = "Pressure Leak Start"
                
        # Current Anomalies
        else:
            if rng.random() > 0.5:
                # Current surge (2x normal)
                df.iloc[idx, df.columns.get_loc("current")] *= 2.0
                anomaly_labels[ts] = "Current Surge"
            else:
                # Intermittent spikes
                spike_len = min(5, n_points - idx)
                for i in range(spike_len):
                    if i % 2 == 0:
                        df.iloc[idx+i, df.columns.get_loc("current")] += rng.normal(10, 2)
                df.iloc[idx:idx+spike_len, df.columns.get_loc("is_anomaly")] = 1
                anomaly_labels[ts] = "Current Spikes Start"
                
        df.iloc[idx, df.columns.get_loc("is_anomaly")] = 1
        
    return df, anomaly_labels

def generate_sensor_data(config: dict) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, str]]:
    """
    Generate synthetic multi-sensor dataset.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        Tuple[pd.DataFrame, Dict[pd.Timestamp, str]]: DataFrame of sensor readings and anomaly labels.
    """
    logger.info("Generating synthetic sensor data...")
    rng = np.random.default_rng(config["model"]["random_state"])
    
    n_points = config["data"]["n_points"]
    freq = config["data"]["frequency"]
    start_date = config["data"]["start_date"]
    
    # Create time index
    timestamps = pd.date_range(start=start_date, periods=n_points, freq=freq)
    df = pd.DataFrame(index=timestamps)
    
    sensors_cfg = config["data"]["sensors"]
    
    # Sensor A: Temperature (daily sinusoidal drift)
    t_cfg = sensors_cfg["temperature"]
    # 24h = 96 periods of 15min
    time_angles = np.linspace(0, (n_points / 96) * 2 * np.pi, n_points)
    daily_drift = t_cfg["drift_amplitude"] * np.sin(time_angles)
    df["temp"] = rng.normal(t_cfg["mu"], t_cfg["sigma"], n_points) + daily_drift
    
    # Sensor B: Vibration (with random micro-bursts)
    v_cfg = sensors_cfg["vibration"]
    base_vib = rng.normal(v_cfg["mu"], v_cfg["sigma"], n_points)
    # Add micro-bursts (small frequent noise)
    micro_bursts = rng.exponential(scale=0.5, size=n_points) * (rng.random(n_points) > 0.95)
    df["vibration"] = base_vib + micro_bursts
    
    # Sensor C: Pressure
    p_cfg = sensors_cfg["pressure"]
    df["pressure"] = rng.normal(p_cfg["mu"], p_cfg["sigma"], n_points)
    
    # Sensor D: Current Draw (correlated with Temperature)
    c_cfg = sensors_cfg["current"]
    base_current = rng.normal(c_cfg["mu"], c_cfg["sigma"], n_points)
    df["current"] = base_current + 0.3 * (df["temp"] - t_cfg["mu"])
    
    # Ensure no negative values where physically improbable
    df["vibration"] = np.maximum(df["vibration"], 0.0)
    df["current"] = np.maximum(df["current"], 0.0)
    
    # Inject Anomalies
    df_anom, labels = inject_anomalies(df, rng, config)
    
    # Reset index to make timestamp a column
    df_anom = df_anom.reset_index().rename(columns={"index": "timestamp"})
    
    logger.info(f"Generated {n_points} data points with {df_anom['is_anomaly'].sum()} anomalous timestamps.")
    return df_anom, labels

def plot_raw_sensors(df: pd.DataFrame, save_path: str) -> None:
    """
    Plot raw sensor data and save to file.

    Args:
        df (pd.DataFrame): Sensor DataFrame.
        save_path (str): Path to save the plot.
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    sensors = ["temp", "vibration", "pressure", "current"]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
    
    for ax, sensor, color in zip(axes, sensors, colors):
        ax.plot(df["timestamp"], df[sensor], label=sensor, color=color, alpha=0.7)
        anomalies = df[df["is_anomaly"] == 1]
        ax.scatter(anomalies["timestamp"], anomalies[sensor], color='red', s=20, label='Anomaly')
        ax.set_ylabel(sensor.capitalize())
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved raw sensor plot to {save_path}")
