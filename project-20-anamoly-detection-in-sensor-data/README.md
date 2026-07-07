# SentinelFlow — AI-Powered Sensor Anomaly Detection

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red.svg)]()

### *"Monitor everything. Miss nothing."*

**SentinelFlow** is a production-grade, unsupervised anomaly detection system for multi-sensor IoT data streams.  
Built with Isolation Forest + engineered time-series features — no labeled anomaly data required.

[📁 Project Structure](#-project-structure)

</div>

---

## 📌 Overview

Industrial equipment fails silently — until it doesn't. SentinelFlow monitors real-time sensor streams across **Temperature, Vibration, Pressure, and Current** channels, flags anomalies the moment they emerge, and delivers actionable reports — all without a single labeled training example.

Whether you're maintaining factory equipment, monitoring server infrastructure, or building a predictive maintenance pipeline, SentinelFlow gives you:

- **Unsupervised detection** — no labeled anomaly data needed
- **Real-time Streamlit dashboard** — upload CSV, get results instantly
- **Automatic PDF evaluation reports** — shareable, professional output
- **Full configurability** via `config.yaml` — zero hardcoded parameters

---

## 🖼️ App Screenshots

### 1. 🏠 Main Dashboard — Live Sensor Overview
> Real-time multi-sensor monitoring with anomaly score timeline

<img width="2556" height="1152" alt="01_main_dashboard png" src="https://github.com/user-attachments/assets/47ed46be-e6ef-4b09-87fa-62cc3b6381c9" />


---

### 2. 📡 Sensor Feed — Raw Data Visualization
> Incoming sensor streams rendered with anomaly markers overlaid
<img width="342" height="1058" alt="02_sensor_feed png" src="https://github.com/user-attachments/assets/a5646f00-379c-4664-8ce7-b5c42034e2a8" />


---

### 3. 🚨 Anomaly Detection Panel
> Detected anomalies highlighted across all channels with timestamps

<img width="2110" height="1108" alt="03_anomaly_detection png" src="https://github.com/user-attachments/assets/ee7f2c5b-a054-44ca-adce-8fd16a5a3667" />


---

### 4. 📈 Feature Engineering View
> Rolling statistics, lag features, and cross-sensor ratio analysis

<img width="2146" height="904" alt="04_feature_engineering png" src="https://github.com/user-attachments/assets/a147dc2a-c1f3-499a-a743-c3839dc5eaad" />


---

### 5. 📂 CSV Upload & Custom Dataset Flow
> Upload your own sensor data and get detection results in seconds

<img width="1870" height="840" alt="05_csv_upload png" src="https://github.com/user-attachments/assets/c73c1693-3c06-4dc1-9e2f-4a7e9bbb7f35" />


---

### 6. 📄 Train Model
> chose hyperparameter , set threshhold ,generate data and train model

<img width="1980" height="912" alt="06_evaluation_report png" src="https://github.com/user-attachments/assets/c971b82b-5ac6-4763-9ca9-f2b18037a3c8" />



---

## ✨ Key Features

| Feature | Details |
|---|---|
| 📡 **Multi-Sensor Support** | Temperature, Vibration, Pressure, Current |
| 🌲 **Isolation Forest** | Unsupervised — no labeled anomalies required |
| 🔧 **Feature Engineering** | Rolling stats, lag features, cross-sensor ratios, Z-scores |
| 📊 **Interactive Dashboard** | Live Plotly anomaly timeline via Streamlit |
| ⚙️ **Zero Hardcoded Values** | Fully configurable via `config.yaml` |
| 📥 **Custom CSV Upload** | Bring your own sensor data for instant analysis |
| 📄 **Auto Report Generation** | PDF-quality evaluation report with metrics |

---

## 🧠 How It Works

```text
╔══════════════╗    ╔══════════════════════╗    ╔════════════════════╗    ╔═══════════════════╗
║  Sensor Data ║───▶║  Feature Engineering ║───▶║  Isolation Forest  ║───▶║  Anomaly Decision ║
╚══════════════╝    ╚══════════════════════╝    ╚════════════════════╝    ╚═══════════════════╝
  4 channels            Rolling mean/std           n_estimators=200          Score threshold
  15-min freq           Lag features (t-1,t-2)     contamination=3%          → ALERT 🚨
  Synthetic or          Cross-sensor ratios         Unsupervised              → NORMAL ✅
  uploaded CSV          Z-score normalization       No labels needed
```

### Why Isolation Forest?

Isolation Forest is ideal for sensor anomaly detection because:
- It works **without any labeled anomaly examples** (which are rare and expensive in real deployments)
- It scales efficiently to high-dimensional, high-frequency sensor data
- It is highly interpretable — anomaly scores map directly to isolation depth
- It handles **multivariate anomalies** across correlated sensor channels

---

## 📊 Results Benchmark

Evaluated on synthetic sensor data with injected ground-truth anomalies:

| Sensor Channel | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|
| **All Sensors (combined)** | 0.91 | 0.87 | 0.89 | 0.94 |
| Temperature | 0.93 | 0.90 | 0.91 | 0.96 |
| Vibration | 0.88 | 0.84 | 0.86 | 0.92 |

> ⚠️ Results vary with real-world data. Use `make train` on your dataset for dataset-specific metrics.

---

## 🚀 Quick Start

> Get SentinelFlow running in under 2 minutes with 4 commands.

```bash
# 1. Clone the repository
git clone https://github.com/hasana157/SentinelFlow && cd SentinelFlow

# 2. Install dependencies
make setup

# 3. Train the Isolation Forest model
make train

# 4. Launch the Streamlit dashboard
make app
```

The dashboard will open at `http://localhost:8501` — ready to monitor.

---

## 🗄️ Using a Custom Kaggle Dataset

SentinelFlow works out of the box with any sensor CSV from Kaggle:

1. **Download** a sensor dataset (e.g., *Predictive Maintenance Dataset*, *Sensor Fault Detection*)
2. **Rename** the CSV to `sensor_data.csv`
3. **Place** it inside the `data/` folder (replaces the synthetic default)
4. **Update `config.yaml`** if your column names differ — or format your CSV to match:
   ```
   timestamp, temp, vibration, pressure, current
   ```
5. Run `make train` → `make app` and you're live

---

## 📁 Project Structure

```
SentinelFlow/
│
├── app/
│   └── dashboard.py          # Streamlit dashboard entry point
│
├── data/
│   ├── sensor_data.csv       # Default synthetic dataset (replaceable)
│   └── raw/                  # Place custom Kaggle datasets here
│
├── models/
│   └── isolation_forest.pkl  # Saved trained model
│
├── src/
│   ├── data_generator.py     # Synthetic sensor data generator
│   ├── feature_engineering.py# Rolling stats, lags, cross-sensor ratios
│   ├── train.py              # Model training pipeline
│   ├── predict.py            # Inference + anomaly scoring
│   └── report.py             # Evaluation report generator
│
├── assets/
│   └── screenshots/          # UI screenshots (linked above)
│
├── config.yaml               # All model + app parameters (no hardcoding)
├── Makefile                  # One-command setup, train, run
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration (`config.yaml`)

All parameters are centralized — no digging through source files:

```yaml
model:
  n_estimators: 200
  contamination: 0.03       # Expected anomaly rate (3%)
  random_state: 42

features:
  rolling_window: 10        # Rolling mean/std window size
  lag_steps: [1, 2, 3]     # Lag feature offsets

data:
  sensors: [temp, vibration, pressure, current]
  frequency: "15min"
  n_samples: 5000

thresholds:
  alert_score: -0.1         # Isolation Forest score cutoff for ALERT
```

---

## 💡 Use Cases

- 🏭 **Predictive Maintenance** — catch equipment faults before failure
- 🖥️ **Server Infrastructure** — CPU/memory/thermal anomaly alerts
- 🌾 **Agricultural IoT** — soil, weather, irrigation sensor monitoring
- 🏥 **Medical Devices** — QA monitoring for life-critical hardware
- ⚡ **SCADA Systems** — power grid and energy infrastructure

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Machine Learning | scikit-learn (Isolation Forest) |
| Data Processing | pandas, numpy |
| Web Application | Streamlit 1.28.0 |
| Visualization | Plotly, Matplotlib, Seaborn |
| Configuration | PyYAML |
| Reporting | FPDF / ReportLab |

---

## 🗺️ Roadmap

- [ ] Real-time MQTT/Kafka stream ingestion
- [ ] SHAP-based anomaly explanation (why flagged?)
- [ ] Email/Slack alerting integration
- [ ] Docker containerization for deployment
- [ ] LSTM-based deep learning comparison module

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

---



## 👩‍💻 Author

**Hasana Zahid**  
AI & ML Engineer | Python Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hasana%20Zahid-blue?logo=linkedin)](https://www.linkedin.com/in/hasana-zahid-605543310)
[![GitHub](https://img.shields.io/badge/GitHub-hasana157-black?logo=github)](https://github.com/hasana157)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-orange?logo=firefox)](https://github.com/hasana157/hasana-zahid_portfolio)

---

<div align="center">

⭐ **If SentinelFlow helped you, consider starring the repo — it helps others find it!** ⭐

</div>
