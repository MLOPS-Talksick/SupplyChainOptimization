SupplySense
AI-powered demand forecasting for modern supply chains

Introduction
SupplySense is an end-to-end, MLOps-driven demand forecasting platform built to address the long-standing inefficiencies in supply chain inventory management. The core idea stems from a simple but critical gap — most supply chains today rely on static ERP-based rule systems or manual spreadsheets to predict demand. These systems fall short when facing real-world volatility like seasonality, market shocks, or dynamic pricing.

In this project, we’ve developed a modular, scalable, and cloud-native machine learning system that forecasts product-level demand with high accuracy. The platform automates the full forecasting lifecycle — from raw data ingestion and preprocessing, through model training and deployment, all the way to real-time prediction and monitoring. It’s built to serve enterprises handling thousands of SKUs across geographies, and it’s designed to be easily integrated with existing retail, e-commerce, or manufacturing systems.

Problem Statement
Global supply chains suffer from a persistent mismatch between inventory and actual demand. This leads to overstocking, stockouts, missed revenue, and bloated operational costs. According to recent reports, companies lose over $1.8 trillion every year due to poor demand-supply alignment. Traditional ERP systems, while widely used, are rigid and reactive — they cannot adapt to complex, fast-changing environments.

Manual forecasting introduces delays and inaccuracies. Businesses struggle to scale these processes or to incorporate external signals (like pricing trends, seasonal behavior, or sudden demand spikes) into their planning. What’s needed is an intelligent, adaptive system that learns from historical and real-time data to produce reliable, actionable demand forecasts.

Vision
SupplySense aims to be the intelligence layer that powers future-ready supply chains. The platform leverages the latest advancements in machine learning, time-series modeling, and cloud automation to bring precision forecasting into operational workflows.

Project Objective
The primary objective of SupplySense is to build an end-to-end system capable of:
Forecasting product demand across thousands of SKUs and regions
Reducing forecast error and its downstream impact on inventory decisions
Automatically retraining and redeploying models as new data arrives
Providing infrastructure that is scalable, portable, and easy to monitor

The platform is designed for integration with existing ERP, warehouse management, or POS systems and can be customized for use in sectors such as retail, FMCG, logistics, and manufacturing.

Methodology and System Overview
SupplySense operates through a modular MLOps pipeline divided into five core stages: data ingestion, data preprocessing, model training, deployment, and monitoring. The architecture is built with automation, scalability, and traceability at its core, enabling the system to be fully reproducible and production-ready.

1. Data Ingestion
Users can upload structured transaction or product-level data into a Google Cloud Storage bucket. This step serves as the entry point to the pipeline. Ingestion DAGs are orchestrated using Apache Airflow, ensuring that all operations are scheduled, monitored, and recoverable in case of failure.

Data versioning is managed using DVC, allowing us to maintain strict control over every dataset used for model training or inference.

2. Data Preprocessing and Validation
Once ingested, data flows through multiple validation and transformation stages. Pre-validation checks ensure schema integrity and missing value detection, followed by cleaning and feature engineering using Pandas. Post-validation confirms the quality of outputs before they are passed on for training.

Both raw and processed data are logged and stored for traceability and auditing.

3. Model Training and Forecasting
The processed data is used to train a set of forecasting models including XGBoost, LSTM, DeepAR, and SARIMA. These models are designed to handle time series forecasting with varying levels of complexity and temporal patterns.

Model training is conducted on Google Vertex AI, enabling scalable, high-performance compute. Parameters are tuned automatically, and model evaluation includes accuracy scoring, drift detection, and version tagging.

The output includes both forecasts and diagnostic metrics, stored in a centralized database.

4. CI/CD and Deployment
The platform is fully containerized using Docker. Every pipeline component is built as a modular container, ensuring that models and services can be deployed consistently across environments.

Terraform is used to provision GCP infrastructure, and GitHub Actions handles CI/CD, including:

Auto-deploying changes on new commits

Running validation tests

Building and pushing Docker images to the Artifact Registry

This enables the system to maintain continuous delivery with minimal manual overhead.

5. Monitoring and Observability
All predictions, logs, and model statistics are stored in a SQL backend. Monitoring dashboards are set up using Grafana (or GCP-native alternatives), enabling teams to track:

Forecast accuracy over time

Data and model drift

Operational pipeline health

Automated alerts via email notify the team in case of anomalies, such as prediction errors crossing a defined threshold or pipeline failures.

Our vision is to enable businesses to move from guesswork to data-driven decision-making — predicting demand at scale, adapting in real time, and optimizing inventory across warehouses, products, and regions.


Technology Stack

Component	Tools & Services
Data Ingestion	Python, Pandas, Google Cloud Storage
Workflow Orchestration	Apache Airflow, DVC
Modeling	XGBoost, DeepAR, LSTM, SARIMA
Model Training & Hosting	Google Vertex AI
Infrastructure	Terraform, Docker
CI/CD	GitHub Actions
Serving	FastAPI, Next.js
Monitoring	SQL, Grafana, Custom Alerting Layer


Why This Matters
Supply chains are moving from static systems to intelligent platforms. SupplySense is a step in that direction — making forecasting smarter, faster, and operationally integrated. With minimal effort, businesses can plug this system into their data stack and start generating demand forecasts that actually reflect market behavior.

This isn't just a model — it's a product-ready forecasting engine designed to solve one of the most expensive inefficiencies in global operations.


---

## Objective

This repository implements an end-to-end **LSTM-based demand forecasting system** with a fully integrated **MLOps architecture**. It automates the lifecycle from raw data ingestion to model health checks, Dockerized deployment, and real-time monitoring. Designed for high-frequency retail or manufacturing data, the system scales effortlessly using **Google Cloud Platform (GCP)** and **Vertex AI**.

---

## Architecture Overview

```
 Raw Sales Data
   └── Data Validation
        └── Data Preprocessing
             └──  LSTM Model Training (Vertex AI)
                  └──  Model Health Check & Diagnostics
                       └──  Dockerized CI/CD Pipeline
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
- **Automated Health Checks:** Validates model accuracy and detects drift using metrics like RMSE, MAPE, and KS-test p-values.
- **Email Notifications:** Alerts stakeholders in real time upon failure or anomalies.
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

## 🔍 Model Monitoring & Validation

| Metric           | Description                                       |
|------------------|---------------------------------------------------|
| `RMSE`           | Root Mean Squared Error for prediction accuracy   |
| `MAPE`           | Mean Absolute Percentage Error                   |
| `KS-Test`        | Data drift detection comparing input distributions |
| `Email Alerts`   | Triggered on threshold violations or drift       |
| `MLflow Logs`    | Track hyperparameters, metrics, and versions     |

---

## ☁️ Cloud Infrastructure

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

## Use Cases

- Inventory management and warehouse stocking
- Retail demand prediction
- Logistics and supply optimization
- Production planning for manufacturing

---

## Future Enhancements


---

---
