
## Backend
```
backend/
├── main.py                        
├── requirements.txt
├── Dockerfile
```
---

**File:** ⁠ `Dockerfile` ⁠  
**Purpose:**  
Defines the containerized environment for deploying a FastAPI application using Python 3.11 on Cloud Run.
**Key Features:**
•⁠  ⁠Uses a lightweight `python:3.11-slim` base image to minimize image size.  
•⁠  ⁠Sets essential environment variables for optimized Python behavior and pip operations.  
•⁠  ⁠Establishes `/app` as the working directory for the container.  
•⁠  ⁠Installs all dependencies listed in `requirements.txt`.  
•⁠  ⁠Copies the entire application codebase into the container.  
•⁠  ⁠Exposes port `8080` to enable traffic from Cloud Run.  
•⁠  ⁠Launches the FastAPI app using `uvicorn` server.
Runs `uvicorn main:app --host 0.0.0.0 --port 8080` to serve the FastAPI app.

---
