##  ML_Models/
```
ML_Models/
├── experiments/
│   ├── deepar.py                  
│   ├── lstm.py                    
│   ├── sarima.py                  
│   └── sarima_adf.py             
├── scripts/
│   ├── health_check.py        
│   ├── model_lstm.py             
│   └── utils.py                   
├── Dockerfile
├── docker-compose.yml             
├── requirements.txt
├── database.env                  

```
### 1. Model Health
**File:** `model_health.py`  
**Purpose:** Performs automatic health checks for the ML model by comparing predictions vs. actuals using statistical metrics, and pushes results to Cloud Monitoring.

**Key Features:**  
- **API Endpoint:** `/model/health` provides a health summary for the last 30 days.  
- **Metrics Logged:** Computes per-product and average **RMSE** and **KS-test p-value**.  
- **Thresholding:** Flags model as *unhealthy* if RMSE > `18.0` or if distribution drift is detected.  
- **Cloud SQL Integration:** Pulls prediction and actual sales data from MySQL via SQLAlchemy.  
- **Observability:** Sends custom metrics (`rmse`, `p_value`) to **Google Cloud Monitoring**.  
- **Logging & Middleware:** Logs incoming requests and health check execution time.  
- **Secure Access:** Token-based authentication for endpoint protection.

```bash
POST /model/health
```

---

### 2. LSTM Model Pipeline  
**File:** `debiased_lstm_pipeline.py`  
**Purpose:** Trains a fairness-aware LSTM model for demand forecasting, mitigating product-level bias and generating actionable insights and explainability artifacts.

**Key Features:**  
- **End-to-End Workflow:** Handles data loading, cleaning, augmentation, feature engineering, model training, evaluation, and artifact management.
- **Data Quality Checks:** Detects missing values, duplicates, negative sales, sparse samples, and large time gaps.
- **Smart Augmentation:** Balances products with few samples via synthetic interpolation and noise.
- **Advanced Feature Engineering:** Generates rolling stats, lag features, holiday indicators, and group statistics per product.
- **Fairness-Aware Training:** Custom callback monitors and mitigates per-product performance disparities during LSTM training.
- **Bias Detection & Correction:** Quantifies and corrects for systematic prediction bias (mean/median error, heteroscedasticity).
- **Optuna Tuning:** Optimizes hyperparameters (units, dropout, activations, learning rate) via automated search.
- **Explainability:** SHAP-based analysis for feature impact and bias visualization.
- **Automated Reporting:** Generates markdown report with fairness and performance insights, and emails results to stakeholders.
- **Artifact Storage:** Uploads trained model, scalers, encoders, SHAP outputs, and plots to Google Cloud Storage.

```bash
python debiased_lstm_pipeline.py
```

---

7. Model Training Utilities  
**File:** `model_training_utils.py`  
**Purpose:** Supports demand forecasting pipeline with data access, feature engineering, bias analysis, notifications, and artifact uploads.

**Key Features:**  
- **Cloud SQL Access:** Connects securely to MySQL using SQLAlchemy + Cloud SQL Connector.  
- **Feature Engineering:** Adds time-based, lag, and rolling features (`lag_1`, `rolling_mean_7`, etc.).  
- **Bias Detection & Visualization:** Analyzes class imbalance, time gaps, seasonality; saves plots.  
- **Email Alerts:** Sends notifications with optional file/DataFrame attachments via Gmail.  
- **GCS Uploads:** Pushes models and plots to Google Cloud Storage with error handling.

```bash
# Utility import example
from model_training_utils import get_latest_data_from_cloud_sql, send_email
```

--- 
