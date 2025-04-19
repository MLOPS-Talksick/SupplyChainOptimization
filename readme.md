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
