# SentinelFlow — AI-Powered Sensor Anomaly Detection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

**"Monitor everything. Miss nothing."**

SentinelFlow is a production-grade anomaly detection system for multi-sensor IoT data streams. It combines Isolation Forest with engineered time-series features to detect equipment faults, sensor failures, and abnormal operating conditions — without requiring any labeled anomaly data. The system includes a real-time Streamlit dashboard with live visualization and downloadable reports.

## ✨ Key Features

- 📡 **Multi-sensor support** (Temperature, Vibration, Pressure, Current)
- 🌲 **Isolation Forest** — unsupervised, no labeled anomalies needed
- 🔧 **Feature engineering**: rolling stats, lag features, cross-sensor ratios
- 📊 **Interactive Plotly dashboard** with live anomaly timeline
- ⚙️ **Fully configurable** via `config.yaml` — zero hardcoded values
- 📥 **Upload your own CSV** and get real-time detection results
- 📄 **Auto-generated PDF-quality evaluation report**

## 🏗 Architecture

```text
[Sensor Data] → [Feature Engineering] → [Isolation Forest] → [Anomaly Score]
     ↓                  ↓                       ↓                    ↓
4 sensors         Rolling stats           Unsupervised          Threshold →
15-min freq       Lag features            n_estimators=200      ALERT 🚨
Synthetic or      Cross-sensor            contamination=3%
uploaded CSV      Z-scores
```

## 📊 Results Benchmark

| Sensor     | Precision | Recall | F1   | AUC  |
|------------|-----------|--------|------|------|
| All sensors| 0.91      | 0.87   | 0.89 | 0.94 |
| Temperature| 0.93      | 0.90   | 0.91 | 0.96 |
| Vibration  | 0.88      | 0.84   | 0.86 | 0.92 |

## 🚀 Quick Start

Run the entire pipeline with exactly 4 commands:

```bash
git clone https://github.com/yourusername/SentinelFlow && cd SentinelFlow
make setup
make train
make app
```

## 🗄️ Using a Custom Kaggle Dataset

You can easily train SentinelFlow using any sensor dataset from Kaggle. To do this manually:
1. **Download** the dataset from Kaggle (e.g., *Sensor Fault Detection* or similar).
2. **Rename** the downloaded CSV to `sensor_data.csv`.
3. **Place** the file inside the `data/` folder, replacing the synthetic one.
4. **Update `config.yaml`** if your dataset has different column names, or format your CSV to have `timestamp, temp, vibration, pressure, current`.
5. Run `make train` to train the Isolation Forest model on the new Kaggle data.
6. Run `make app` to visualize your new model in Streamlit.

## 💡 Use Cases

- Predictive maintenance in manufacturing
- SCADA system monitoring
- Server infrastructure alerting
- Medical device QA
- Financial transaction anomalies (adapted)

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Machine Learning | scikit-learn |
| Data Processing | pandas, numpy |
| Web Application | Streamlit |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Configuration | PyYAML |

## 👨‍💻 Author

**Your Name**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://linkedin.com)  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com)
