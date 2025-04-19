# Supply Chain Optimization

> **Smarter Demand. Leaner Supply. 
> Forecasting product demand isn't just about predicting numbers—it's about enabling intelligent decisions across your supply chain. This project delivers a production-ready, cloud-deployable LSTM pipeline backed by a complete MLOps stack for real-time business impact.

## Project Overview

This repository implements an end-to-end **LSTM-based demand forecasting system** with a fully integrated **MLOps architecture**. It automates the lifecycle from raw data ingestion to model health checks, Dockerized deployment, and real-time monitoring. Designed for high-frequency retail or manufacturing data, the system scales effortlessly using **Google Cloud Platform (GCP)** and **Vertex AI**.

Architecture Overview

```
Raw Sales Data
   └── Data Validation
        └── Data Preprocessing
             └── LSTM Model Training (Vertex AI)
                  └── Model Health Check & Diagnostics
                       └── Dockerized CI/CD Pipeline
                            └── Monitoring + Email Alerts
```

## Project Structure

```
SupplyChainOptimization/
├── ML_Models/
│   ├── lstm_pipeline.py                # Core model training + prediction logic
│   └── model_health_check_api.py       # Health checks, drift detection, metrics logging
├── data/                               # Input datasets (e.g., Excel/CSV)
├── .github/workflows/ci_cd.yaml        # GitHub Actions pipeline for CI/CD
├── Dockerfile                          # Containerization for deployment
├── requirements.txt                    # Python dependencies
└── README.md

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MLOPS-Talksick/SupplyChainOptimization.git
cd SupplyChainOptimization
```

### 2. Set Up Environment

```bash
pip install -r requirements.txt
```

### 3. Train the LSTM Model

```bash
python ML_Models/lstm_pipeline.py
```

### 4. Run the Model Health Check

```bash
python ML_Models/model_health_check_api.py
```

---

## Model Monitoring & Validation

| Metric           | Description                                       |
|------------------|---------------------------------------------------|
| `RMSE`           | Root Mean Squared Error for prediction accuracy   |
| `MAPE`           | Mean Absolute Percentage Error                   |
| `KS-Test`        | Data drift detection comparing input distributions |
| `Email Alerts`   | Triggered on threshold violations or drift       |
| `MLflow Logs`    | Track hyperparameters, metrics, and versions     |

---

## Cloud Infrastructure

| Component        | Tool                     |
|------------------|--------------------------|
| Cloud Platform   | Google Cloud Platform (GCP) |
| Training & Hosting | Vertex AI               |
| Storage          | Google Cloud Storage (GCS) |
| CI/CD            | GitHub Actions + Docker  |
| Monitoring       | Custom Dashboards + Email Alerts |

---

## Tech Stack

- **Programming**: Python 3.9+
- **ML Framework**: TensorFlow (LSTM)
- **Pipeline**: MLflow, Pandas, Scikit-learn
- **MLOps**: Docker, Vertex AI, GitHub Actions
- **Monitoring**: SciPy (KS-Test), Email Notifier
- **Data Format**: Excel (`.xlsx`) or CSV

---

##  Use Cases

- Inventory management and warehouse stocking
- Retail demand prediction
- Logistics and supply optimization
- Production planning for manufacturing

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

- Yahoo Finance API for providing commodity data
- Apache Airflow for workflow orchestration
- Docker for containerization
