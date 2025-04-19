eCommerce-Chatbot using LangGraph (Verta)
Introduction
The Verta Chatbot project is a sophisticated AI-driven solution designed to enhance user interactions with product information. Deployed as a serverless FASTAPI API on Cloud Run, it serves users by answering questions about products, leveraging both metadata and user reviews for context. The chatbot utilizes a multi-agent workflow, with distinct agents performing specialized roles. A Metadata Agent is responsible for summarizing product descriptions, while a Retriever Agent fetches relevant data from a vector store containing user reviews. This allows the chatbot to answer a wide range of user inquiries, drawing on both product details and feedback from other customers.

The database for this solution is hosted on PostgreSQL in Google Cloud Platform (GCP), ensuring scalable, reliable storage. The project utilizes CI/CD pipelines via GitHub Actions, automating code deployment and ensuring seamless integration and delivery. Additionally, the system incorporates LLM-as-Judge to generate synthetic test questions for a set of products, while bias detection algorithms analyze potential biases in the chatbot's responses to ensure fairness and accuracy.

Experiment tracking is managed through MLflow, which captures model performance and experiment metadata, while Langfuse is used for tracing user interactions and gathering feedback to continuously improve the system. For monitoring and alerting, GCP Logs are utilized with notifications configured to send alerts to a Teams channel for real-time system health checks.

The chatbot’s data orchestration is powered by Apache Airflow, while FaisDB is used as the vector store for storing product reviews and context. The system integrates three LLMs — GPT-4o-Mini, Llama 3.1-70B, and Llama 3.1-8B — running on four nodes to support its operations. The chatbot’s multi-agent flow is managed using LangGraph, a framework for orchestrating complex workflows. For ease of use, the chatbot is also available as a Streamlit web app, with integration capabilities for custom frontends via API, secured using Auth Bearer Tokens.

Project Architecture
Verta Achitecture

Getting Started - Guide
To start working with the ecom-chatbot project, please follow the setup instructions outlined in the Project Setup Guide.

This guide includes steps for creating a GCP account, configuring environment variables, setting up GitHub secrets, deploying locally, and hosting the Chatbot API. Once the initial setup is complete, you can explore the following detailed guides for specific aspects of the project:

Understand the API Payloads
Explore ML Pipelines
Stage 1 - Base Model
Stage 2 - Test Data Ingestion
Stage 3 - Model Evaluation
Stage 4 - Bias Detection
CI/CD Workflow
Cost Analysis
Logging and Monitoring Setup
Version Rollback
Understand the Project Folder Structure


#  Project Architecture – SupplyChainOptimization

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


