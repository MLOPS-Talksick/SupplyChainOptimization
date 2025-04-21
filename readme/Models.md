## Models

The `ML_Models/` directory contains forecasting models, training pipelines, and supporting scripts for automated evaluation, fairness analysis, and deployment.

### 1. Model Variants
Located in `ML_Models/experiments/`:
- `lstm.py`: Standard LSTM model for demand forecasting.
- `deepar.py`: DeepAR-based probabilistic forecasting model.
- `sarima.py` & `sarima_adf.py`: Classical SARIMA models with and without stationarity tests.

### 2. Debiased LSTM Pipeline
File: `debiased_lstm_pipeline.py`  
Purpose: Trains a fairness-aware LSTM model with built-in bias mitigation and explainability.

**Key Features:**
- End-to-end workflow: data cleaning, augmentation, feature engineering, model training, evaluation, artifact management.
- Data quality checks: handles missing values, sparse samples, outliers, and duplicates.
- Augmentation: synthetic sample generation for underrepresented products.
- Fairness-aware training: mitigates per-product performance disparities via custom callback.
- Bias correction: adjusts for systematic prediction bias using post-processing.
- Explainability: SHAP-based feature impact analysis and fairness visualization.
- Reporting: Auto-generates markdown reports on model performance and fairness.
- Artifact storage: Uploads trained model, SHAP values, encoders, and plots to Google Cloud Storage.

### 3. Bias & Explainability Note
The LSTM model is designed to ensure fairness across all product categories. Bias is monitored and corrected using:
- RMSE and MAPE disparity ratios
- SHAP values for interpretability
- Post-training bias correction methods
- Fairness-aware callbacks during training

A summary report is generated with metrics such as:
- RMSE: 2.50
- MAPE: 38.52%
- Fairness metrics: RMSE disparity ratio = 1.00, MAPE disparity ratio = 1.00

### 4. Model Health Monitoring
File: `health_check.py`  
Purpose: Validates model predictions via statistical checks and exposes results via API.

**Key Features:**
- API endpoint: `/model/health` returns 30-day model diagnostics.
- Metrics: RMSE, KS-test p-values computed per product.
- Thresholds: Flags model as unhealthy if RMSE > 18.0 or drift is detected.
- Observability: Sends custom metrics to Google Cloud Monitoring.
- Security: Token-based authentication is enabled.

### 5. Supporting Utilities
File: `model_training_utils.py`  
Provides reusable functions for:
- Cloud SQL access via SQLAlchemy
- Feature engineering (lags, rolling stats, holidays, etc.)
- Bias detection and visualization
- Email alerts with attachments
- Artifact upload to GCS

Example:
```python
from model_training_utils import get_latest_data_from_cloud_sql, send_email
