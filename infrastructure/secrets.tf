resource "google_secret_manager_secret" "postgres_password" {
  secret_id = "postgres_password"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "postgres_password_version" {
  secret      = google_secret_manager_secret.postgres_password.id
  secret_data = var.postgres_password
}

resource "google_secret_manager_secret" "redis_password" {
  secret_id = "redis_password"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "redis_password_version" {
  secret      = google_secret_manager_secret.redis_password.id
  secret_data = var.redis_password
}

resource "google_secret_manager_secret" "airflow_fernet_key" {
  secret_id = "airflow_fernet_key"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_fernet_key_version" {
  secret      = google_secret_manager_secret.airflow_fernet_key.id
  secret_data = var.airflow_fernet_key
}
