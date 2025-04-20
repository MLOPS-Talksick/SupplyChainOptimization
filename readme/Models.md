
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

Let me know if you want even more concise, or if you want a section breakdown for each major function/module!
