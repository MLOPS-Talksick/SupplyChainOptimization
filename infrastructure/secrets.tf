# Postgres User
resource "google_secret_manager_secret" "postgres_user" {
  secret_id = "postgres_user"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "postgres_user_version" {
  secret      = google_secret_manager_secret.postgres_user.id
  secret_data = var.postgres_user

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Postgres Password
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

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Postgres DB
resource "google_secret_manager_secret" "postgres_db" {
  secret_id = "postgres_db"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "postgres_db_version" {
  secret      = google_secret_manager_secret.postgres_db.id
  secret_data = var.postgres_db

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Database Password
resource "google_secret_manager_secret" "airflow_database_password" {
  secret_id = "airflow_database_password"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_database_password_version" {
  secret      = google_secret_manager_secret.airflow_database_password.id
  secret_data = var.airflow_database_password

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Redis Password
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

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Fernet Key
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

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Admin Username
resource "google_secret_manager_secret" "airflow_admin_username" {
  secret_id = "airflow_admin_username"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_admin_username_version" {
  secret      = google_secret_manager_secret.airflow_admin_username.id
  secret_data = var.airflow_admin_username

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Admin Password
resource "google_secret_manager_secret" "airflow_admin_password" {
  secret_id = "airflow_admin_password"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_admin_password_version" {
  secret      = google_secret_manager_secret.airflow_admin_password.id
  secret_data = var.airflow_admin_password

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Admin Firstname
resource "google_secret_manager_secret" "airflow_admin_firstname" {
  secret_id = "airflow_admin_firstname"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_admin_firstname_version" {
  secret      = google_secret_manager_secret.airflow_admin_firstname.id
  secret_data = var.airflow_admin_firstname

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Admin Lastname
resource "google_secret_manager_secret" "airflow_admin_lastname" {
  secret_id = "airflow_admin_lastname"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_admin_lastname_version" {
  secret      = google_secret_manager_secret.airflow_admin_lastname.id
  secret_data = var.airflow_admin_lastname

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow Admin Email
resource "google_secret_manager_secret" "airflow_admin_email" {
  secret_id = "airflow_admin_email"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_admin_email_version" {
  secret      = google_secret_manager_secret.airflow_admin_email.id
  secret_data = var.airflow_admin_email

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Airflow UID
resource "google_secret_manager_secret" "airflow_uid" {
  secret_id = "airflow_uid"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "airflow_uid_version" {
  secret      = google_secret_manager_secret.airflow_uid.id
  secret_data = var.airflow_uid

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Docker GID
resource "google_secret_manager_secret" "docker_gid" {
  secret_id = "docker_gid"
  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "docker_gid_version" {
  secret      = google_secret_manager_secret.docker_gid.id
  secret_data = var.docker_gid

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret" "project_id" {
  secret_id = "project_id"
  replication {
    user_managed {
      replicas { location = "us-central1" }
    }
  }
}

resource "google_secret_manager_secret_version" "project_id_version" {
  secret      = google_secret_manager_secret.project_id.id
  secret_data = var.project_id

  lifecycle {
    ignore_changes = [secret_data]
  }
}