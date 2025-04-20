## Architecture Modules

### Data Ingestion
- Collects raw sales data via upload or system integration.
- Applies schema validation and consistency checks.
- Stores validated data in cloud storage for downstream use.

### Data Processing
- Orchestrated using Airflow DAGs.
- Performs pre-validation, preprocessing (cleaning, feature engineering), and post-validation.
- Outputs cleaned data to a structured database and versions it using DVC.

### Model Development
- Trains an LSTM forecasting model on preprocessed data using Vertex AI.
- Saves model artifacts and evaluation metrics.
- Deploys the trained model to an API or Vertex AI Endpoint.
- Includes automated health checks to validate model quality before production use.

### CI/CD Deployment
- Uses GitHub Actions to automate testing, Dockerization, and deployment.
- Manages infrastructure via Terraform for consistent cloud resource provisioning.

### Monitoring & Logging
- Continuously monitors model performance (e.g., RMSE, MAPE) and data drift.
- Sends alerts on anomalies or performance degradation.
- Captures detailed logs for auditing and troubleshooting.
