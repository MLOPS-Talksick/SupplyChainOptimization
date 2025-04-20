## Architecture 

<p align="center">
  <img src="/Media/arch1.png" alt="Architecture" width="1000"/>
</p>


### Data Ingestion
- Collects raw sales data via upload or system integration.
- Applies schema validation and consistency checks.
- Stores validated data in cloud storage for downstream use.

### Data Validation & Processing
- Orchestrated using Airflow DAGs.
- Performs pre-validation, preprocessing (cleaning, feature engineering).
- Outputs cleaned data to a structured database and versions it using DVC.

### Model Development
- Trains an LSTM forecasting model on preprocessed data using Vertex AI.
- Saves model artifacts and evaluation metrics.
- Deploys the trained model to an API.
- Includes automated health checks to validate model quality before production use.

### CI/CD Deployment
- Uses GitHub Actions for CI/CD.
- Manages infrastructure via Terraform for consistent cloud resource provisioning.

### Monitoring & Logging
- Continuously monitors model performance (e.g., RMSE, MAPE) and data drift using K-S test.
- Sends alerts on anomalies or performance degradation.
- Captures detailed logs for auditing and troubleshooting.

## Key Features

- **LSTM Demand Forecasting Model**  
  Utilizes an LSTM neural network designed for time-series forecasting to capture seasonality, trends, and sudden demand changes for improved accuracy.

- **Vertex AI Integration**  
  Enables scalable model training and deployment on Google Cloud using Vertex AI, supporting experiment tracking, versioning, and built-in monitoring.

- **CI/CD with Docker and GitHub Actions**  
  Automates testing, containerization, and deployment of the pipeline using GitHub Actions and Docker for consistency across environments.

- **Automated Data Validation**  
  Applies schema checks, null detection, and range validation on incoming data to ensure quality before processing and training.

- **Airflow-Orchestrated ETL**  
  Uses Apache Airflow DAGs to automate data ingestion, preprocessing, and scheduling of model retraining workflows.

- **MLflow Experiment Tracking**  
  Logs model parameters, metrics, and artifacts using MLflow to enable version comparison, reproducibility, and experiment management.

- **Model Health Checks**  
  Performs RMSE, MAPE evaluation, and KS test for data drift to ensure model quality before deployment and during production monitoring.

- **Monitoring Dashboards**  
  Tracks key performance metrics and data quality indicators over time through integrated dashboards and cloud monitoring tools.

- **Alerting & Notifications**  
  Sends real-time alerts via email or other channels when thresholds are breached or anomalies are detected.

- **Scalable, Modular Infrastructure**  
  Built with modular components deployed on GCP, using Terraform for reproducibility and scalability across data, model, and service layers.
