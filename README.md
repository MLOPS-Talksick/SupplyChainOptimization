# SupplyChainOptimization

> **Smarter Demand. Leaner Supply.**  
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

#  Project Structure

##  Data_Pipeline/
```
Data_Pipeline/
├── scripts/
│   ├── __init__.py                # Python package marker
│   ├── dvc_versioning.py          # DVC-based data versioning
│   ├── logger.py                  # Logger config for pipeline modules
│   ├── post_validation.py         # Post-cleaning validations
│   ├── pre_validation.py          # Pre-cleaning checks (e.g., schema, nulls)
│   ├── preprocessing.py           # Feature engineering and transformation
│   └── utils.py                   # Common utilities
├── tests/
│   ├── __init__.py
│   ├── requirements-test.txt      # Test dependencies
│   ├── testDataPreprocessing.py
│   ├── testDvcVersioning.py
│   ├── testPostValidation.py
│   ├── testPreValidation.py
│   └── testUtils.py
├── Dockerfile                     # Container for running pipeline
├── requirements.txt               # Runtime dependencies
└── README.md
```

---

##  ML_Models/
```
ML_Models/
├── experiments/
│   ├── deepar.py                  # DeepAR time series model
│   ├── lstm.py                    # LSTM forecasting model
│   ├── sarima.py                  # SARIMA classical forecasting
│   └── sarima_adf.py              # SARIMA with stationarity test (ADF)
├── scripts/
│   ├── model_monitoring.py        # Drift detection, performance tracking
│   ├── model_prediction.py        # Model inference endpoints
│   ├── model_uploader.py          # Push models to registry (e.g., GCP)
│   ├── model_xgboost.py           # Baseline XGBoost training
│   └── utils.py                   # Evaluation & helper functions
├── Dockerfile
├── docker-compose.yml             # Local service orchestration
├── requirements.txt
├── database.env                   # DB connection secrets
└── README.md
```

---

## backend/
```
backend/
├── main.py                        # API logic and routes
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## frontend/
```
frontend/
├── components/
│   └── ui/                        # ShadCN / shared UI blocks
├── lib/                           # Helpers for fetching/config
├── public/                        # Static files
├── src/                           # Core app logic and pages
├── .gitignore
├── README.md
├── components.json
├── eslint.config.mjs
├── next.config.ts
├── package-lock.json
├── package.json
├── postcss.config.mjs
└── tsconfig.json
```

---

## infrastructure/
```
infrastructure/
├── artifact_registry.tf           # GCP Artifact Registry for models/images
├── autoscaler.tf                  # VM autoscaling logic
├── compute_instance.tf            # Base VM instance config
├── compute_instance_baker.tf      # Image baker (custom GCP images)
├── compute_instance_template.tf   # Template for reproducible VM setups
├── firewall.tf                    # Network access rules
├── instance_group.tf              # Instance group definition
├── load_balancer.tf               # HTTPS load balancer config
├── network.tf                     # VPC, subnet setup
├── outputs.tf                     # Exported values like IPs, URLs
├── provider.tf                    # GCP provider config
├── ssh_key.tf                     # Inject SSH keys for access
└── variables.tf                   # TF variables for reusability
```

---

## dags/
```
dags/
├── data_pipeline_dag.py           # Orchestration of ingestion → preprocessing
├── model_pipeline_dag.py          # Training → Evaluation → Monitoring
```

---

## Root-Level Files
```
├── docker-compose.yaml            # Spins up entire stack locally
├── .env                           # Local env vars
├── pyproject.toml                 # Formatter, linter settings
├── .flake8                        # Code style rules
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
