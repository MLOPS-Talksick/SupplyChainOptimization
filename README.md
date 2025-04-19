# SupplyChainOptimization

> **Smarter Demand. Leaner Supply. Powered by LSTM and MLOps.**  
> Forecasting product demand isn't just about predicting numbers—it's about enabling intelligent decisions across your supply chain. This project delivers a production-ready, cloud-deployable LSTM pipeline backed by a complete MLOps stack for real-time business impact.

---

## Objective

This repository implements an end-to-end **LSTM-based demand forecasting system** with a fully integrated **MLOps architecture**. It automates the lifecycle from raw data ingestion to model health checks, Dockerized deployment, and real-time monitoring. Designed for high-frequency retail or manufacturing data, the system scales effortlessly using **Google Cloud Platform (GCP)** and **Vertex AI**.

---

## Architecture Overview

```
Raw Sales Data
   └── Data Validation
        └── Data Preprocessing
             └── LSTM Model Training (Vertex AI)
                  └── Model Health Check & Diagnostics
                       └── Dockerized CI/CD Pipeline
                            └── Monitoring + Email Alerts
```

---

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
```

---

## Key Features

- **LSTM Forecasting:** Deep learning model tailored for time series prediction of product demand.
- **Robust Data Handling:** Includes pre-validation, preprocessing, and post-cleaning checks.
- **Productionized Model:** Packaged using Docker and deployed via CI/CD on GCP.
-**Automated Health Checks:** Validates model accuracy and detects drift using metrics like RMSE, MAPE, and KS-test p-values.
-  **Email Notifications:** Alerts stakeholders upon failure or anomalies in real time.
- **Scalable Training:** Leveraged via **Vertex AI** for on-demand model retraining.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MLOPS-Talksick/SupplyChainOptimization.git
cd SupplyChainOptimization
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
touch .env
```

Edit the `.env` file and add:

```env
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your/cloud_run.json
MLFLOW_TRACKING_URI=http://localhost:5001
GCP_PROJECT=your-gcp-project-id
BUCKET_NAME=your-gcs-bucket-name
```

(Optional) Start MLflow Tracking Server:

```bash
mlflow ui --port 5001
```

(Optional) Authenticate with GCP CLI:

```bash
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project $GCP_PROJECT
```

(Optional) Build and run Docker container:

```bash
docker build -t supplychain-lstm .
docker run -p 8080:8080 supplychain-lstm
```

---

### 3. Train the LSTM Model

```bash
python ML_Models/lstm_pipeline.py
```

---

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

##  Cloud Infrastructure

| Component        | Tool                     |
|------------------|--------------------------|
| Cloud Platform   | Google Cloud Platform (GCP) |
| Training & Hosting | Vertex AI               |
| Storage          | Google Cloud Storage (GCS) |
| CI/CD            | GitHub Actions + Docker  |
| Monitoring       | Custom Dashboards + Email Alerts |

---

##  Tech Stack

- **Programming**: Python 3.9+
- **ML Framework**: TensorFlow (LSTM)
- **Pipeline**: MLflow, Pandas, Scikit-learn
- **MLOps**: Docker, Vertex AI, GitHub Actions
- **Monitoring**: SciPy (KS-Test), Email Notifier
- **Data Format**: Excel (`.xlsx`) or CSV

---

## Use Cases

- Inventory management and warehouse stocking
- Retail demand prediction
- Logistics and supply optimization
- Production planning for manufacturing

---

## Future Enhancements

- [ ] Streamlit/Gradio dashboard for interactive forecasting
- [ ] Real-time ingestion using Kafka or GCP Pub/Sub
- [ ] Hyperparameter optimization with Optuna or Keras Tuner
- [ ] Integration with BigQuery for automated dataset pulls

---



## License
 See `LICENSE` for more details.
