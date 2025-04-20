
## Backend
```
backend/
├── main.py                        
├── requirements.txt
├── Dockerfile
```
---

### 2. Docker Configuration  
**File:** `Dockerfile`  
**Purpose:** Builds a lightweight, production-ready container for deploying a FastAPI application on Cloud Run.  

**Key Features:**  
- **Base Image:** Uses `python:3.11-slim` to keep the image lean and efficient.  
- **Environment Setup:** Configures Python and pip environment variables for optimal performance and cache control.  
- **App Structure:** Sets `/app` as the working directory and copies all source files into it.  
- **Dependency Installation:** Installs project dependencies from `requirements.txt` without caching.  
- **Port Exposure:** Opens port `8080` to allow external access via Cloud Run.  
- **App Execution:** Launches the FastAPI application using `uvicorn` for asynchronous HTTP serving.

```bash
# Build and run locally
docker build -t fastapi-app .
docker run -p 8080:8080 fastapi-app
```

---
