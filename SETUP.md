## Project Setup

This README guides you through setting up the project from scratch, including GCP configuration and GitHub Actions secrets.

---

### 1. Prerequisites

- **Google Cloud Platform (GCP) account** with billing enabled.
- **GitHub repository** for this project.
- **gcloud CLI** installed locally (optional, for manual verification).
- **Docker** installed (required for container builds).

---

### 2. GCP Configuration

1. **Create a new GCP project**
   - Go to the [GCP Console](https://console.cloud.google.com/).
   - Click **Select a project** → **New Project**.
   - Enter a name and note the **Project ID**.

2. **Enable required APIs**
   - Navigate to **APIs & Services** → **Library**.
   - Enable:
     - Artifact Registry API
     - Cloud Storage JSON API
     - IAM API
     - Cloud Key Management Service API (if using KMS for Fernet keys)

3. **Create a Service Account**
   - Go to **IAM & Admin** → **Service Accounts**.
   - Click **+ Create Service Account**.
   - Enter a name and ID (e.g., `deployment-sa`).

4. **Assign roles**
   Grant the following roles to the service account:
   - `Artifact Registry Administrator`
   - `Artifact Registry Reader`
   - `IAM OAuth Client Admin`
   - `Project IAM Admin`
   - `Service Account Admin`
   - `Service Account Key Admin`
   - `Storage Admin`
   - `Storage Bucket Viewer`

5. **Generate and download a key**
   - In the Service Account details, go to **Keys** → **Add Key** → **Create New Key**.
   - Select JSON and download the key file.
   - Store this file securely; you'll reference it in GitHub Actions as `BOOTSTRAP_GCP_KEY`.

---

### 3. Configure GitHub Actions Secrets

Navigate to your GitHub repository's **Settings** → **Secrets and variables** → **Actions**. Create the following secrets with these exact values (or replace placeholders where indicated):

| Secret Name               | Value                                    |
|---------------------------|------------------------------------------|
| AIRFLOW_ADMIN_EMAIL       | `<your_email@example.com>`               |
| AIRFLOW_ADMIN_FIRSTNAME   | `Admin`                                  |
| AIRFLOW_ADMIN_LASTNAME    | `User`                                   |
| AIRFLOW_ADMIN_PASSWORD    | `admin`                                  |
| AIRFLOW_ADMIN_USERNAME    | `admin`                                  |
| AIRFLOW_DATABASE_PASSWORD | `admin`                                  |
| AIRFLOW_DAG_ID            | `gcp_processing_on_demand`               |
| AIRFLOW_FERNET_KEY        | `<your_fernet_key>`                      |
| AIRFLOW_UID               | `50000`                                  |
| API_TOKEN                 | `backendapi1234567890`                   |
| BOOTSTRAP_GCP_KEY         | `<paste_service_account_JSON>`           |
| BUCKET_URI                | `gs://model_training_1`                  |
| DOCKER_GID                | `1000`                                   |
| GCP_PROJECT_ID            | `<your-project-id>`                      |
| GCS_BUCKET_NAME           | `full_raw_data`                          |
| MODEL_NAME                | `lstm_model.keras`                       |
| MYSQL_DATABASE            | `combined_transactions_data`             |
| MYSQL_HOST                | `<your_mysql_host>`                      |
| MYSQL_PASSWORD            | `<your_mysql_password>`                  |
| MYSQL_USER                | `<your_mysql_user>`                      |
| POSTGRES_DB               | `airflow`                                |
| POSTGRES_PASSWORD         | `airflow`                                |
| POSTGRES_USER             | `airflow`                                |
| PROJECT_NUMBER            | `<your_project_number>`                  |
| REDIS_PASSWORD            | `redispass`                              |
| VERTEX_REGION             | `us-central1`                            |

> **Note:** Secrets wrapped in `< >` must be replaced with your specific values; all other values should be set exactly as shown.

---

### 4. Cloning and Initial Deployment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-org>/<your-repo>.git
   cd <your-repo>
   ```

2. **Trigger GitHub Actions**:
   - Push to the `main` branch or open a pull request.
   - The CI/CD pipeline will:
     1. Build and push Docker images to Artifact Registry.
     2. Deploy infrastructure via Terraform.
     3. Deploy Airflow and related services.

3. **Verify deployment**:
   - Check the Actions logs for successful runs.
   - Access the Airflow UI at the provided endpoint.
   - Call the `/health` endpoint or use the CLI to confirm inference is working.

---

### 5. Troubleshooting & Support

- Double-check that all secrets are correctly added and up to date.
- Review GitHub Actions logs for any errors.
- Confirm the service account has been granted all required roles in GCP.
- Use the `gcloud` CLI to inspect IAM policies and service account permissions.


