---

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
