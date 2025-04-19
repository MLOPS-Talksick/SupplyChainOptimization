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

## .github/workflows/
```
.github/workflows/
├── ci.yml                         # Linting, testing, DVC check
├── deploy.yml                     # Terraform + GCP deployment pipeline
```

---

## Root-Level Files
```
├── docker-compose.yaml            # Spins up entire stack locally
├── .env                           # Local env vars
├── pyproject.toml                 # Formatter, linter settings
├── .flake8                        # Code style rules
```



# 🧩 Project Architecture – [SupplyChainOptimization](https://github.com/MLOPS-Talksick/SupplyChainOptimization)

---

## 📁 [Data_Pipeline](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Data_Pipeline)
```
Data_Pipeline/
├── [scripts](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Data_Pipeline/scripts)/
│   ├── dvc_versioning.py
│   ├── logger.py
│   ├── post_validation.py
│   ├── pre_validation.py
│   ├── preprocessing.py
│   └── utils.py
├── [tests](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Data_Pipeline/tests)/
│   ├── requirements-test.txt
│   ├── testDataPreprocessing.py
│   ├── testDvcVersioning.py
│   ├── testPostValidation.py
│   ├── testPreValidation.py
│   └── testUtils.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📁 [ML_Models](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/ML_Models)
```
ML_Models/
├── [experiments](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/ML_Models/experiments)/
│   ├── deepar.py
│   ├── lstm.py
│   ├── sarima.py
│   └── sarima_adf.py
├── [scripts](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/ML_Models/scripts)/
│   ├── model_monitoring.py
│   ├── model_prediction.py
│   ├── model_uploader.py
│   ├── model_xgboost.py
│   └── utils.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── database.env
└── README.md
```

---

## 📁 [backend](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/backend)
```
backend/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📁 [frontend](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend)
```
frontend/
├── [components](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend/components)/
│   └── [ui](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend/components/ui)/
├── [lib](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend/lib)/
├── [public](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend/public)/
├── [src](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend/src)/
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

## 📁 [infrastructure](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/infrastructure)
```
infrastructure/
├── artifact_registry.tf
├── autoscaler.tf
├── compute_instance.tf
├── compute_instance_baker.tf
├── compute_instance_template.tf
├── firewall.tf
├── instance_group.tf
├── load_balancer.tf
├── network.tf
├── outputs.tf
├── provider.tf
├── ssh_key.tf
└── variables.tf
```

---

## 📁 [dags](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/dags)
```
dags/
├── data_pipeline_dag.py
├── model_pipeline_dag.py
```

---

## 📁 [.github/workflows](https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/.github/workflows)
```
.github/workflows/
├── ci.yml
├── deploy.yml
```

---

## 📁 Root-Level Files
```
├── [.env](https://github.com/MLOPS-Talksick/SupplyChainOptimization/blob/main/.env)
├── [docker-compose.yaml](https://github.com/MLOPS-Talksick/SupplyChainOptimization/blob/main/docker-compose.yaml)
├── [pyproject.toml](https://github.com/MLOPS-Talksick/SupplyChainOptimization/blob/main/pyproject.toml)
├── [.flake8](https://github.com/MLOPS-Talksick/SupplyChainOptimization/blob/main/.flake8)
```
