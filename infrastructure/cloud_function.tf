# 1. Upload your Cloud Function source code (ZIP file) to the bucket.
resource "google_storage_bucket_object" "cloud_function_zip" {
  name   = "cloud_function_source.zip"                                 # Object name in the bucket
  bucket = google_storage_bucket.buckets["fully-processed-data"].name # Reference to your bucket using for_each key
  source = "${path.module}/../Cloudrun_Function/GCS_TO_SQL.zip"

  depends_on = [
    google_storage_bucket.buckets["fully-processed-data"]
  ]
}

# 2. Deploy the Cloud Function with an event trigger on the bucket.
resource "google_cloudfunctions2_function" "process_data_function" {
  name        = "processDataFunction"
  location    = var.region
  description = "Triggered when a file is ingested into fully-processed-data bucket"

  build_config {
    runtime     = "python39"
    entry_point = "hello_gcs"
    source {
      storage_source {
        bucket = google_storage_bucket.buckets["fully-processed-data"].name
        object = google_storage_bucket_object.cloud_function_zip.name
      }
    }

    
  }

  service_config {
    available_memory = "512M"
    environment_variables = {
      PROJECT_ID         = var.project_id
      REGION             = var.region
      MODEL_TRAINING_IMAGE_URI = local.model_training_image_uri
      BUCKET_URI         = var.bucket_uri
      MYSQL_HOST         = local.mysql_host
      MYSQL_USER         = var.mysql_user
      MYSQL_DATABASE     = var.mysql_database
      INSTANCE_CONN_NAME = local.instance_conn_name
      MYSQL_PASSWORD     = var.mysql_password
      TRIGGER_TRAINING_URL = "${google_cloud_run_v2_service.model_training_trigger.uri}/trigger-training"

    }
    vpc_connector   = google_vpc_access_connector.cloudrun_connector.id
  }

  event_trigger {
    event_type = "google.cloud.storage.object.v1.finalized"

    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.buckets["fully-processed-data"].name
    }
  }

  depends_on = [
    google_storage_bucket.buckets["fully-processed-data"],
    google_storage_bucket_object.cloud_function_zip
  ]
}


resource "google_cloud_run_v2_service" "model_serving" {
  name     = "model-serving"
  location = var.region
  project  = var.project_id

  template {
    containers {
      # image = "us-central1-docker.pkg.dev/${var.project_id}/${var.artifact_registry}/model_serving:latest"
      image = local.model_serving_image_uri

      env {
        name  = "MYSQL_HOST"
        value = local.mysql_host
      }

      env {
        name  = "MYSQL_USER"
        value = var.mysql_user
      }

      env {
        name  = "MYSQL_PASSWORD"
        value = var.mysql_password
      }

      env {
        name  = "MYSQL_DATABASE"
        value = var.mysql_database
      }

      env {
        name  = "MODEL_NAME"
        value = var.model_name
      }

      env {
        name  = "IMAGE_TAG_HASH"
        value = local.model_serving_image_uri
      }

      env {
        name  = "INSTANCE_CONN_NAME"
        value = local.instance_conn_name
      }


      resources {
        limits = {
          memory = "32Gi"
          cpu    = "8"
        }
      }
    }

    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.id
      egress    = "ALL_TRAFFIC"
    }

    service_account = var.service_account_email

    max_instance_request_concurrency = 80  # optional: adjust if needed
    timeout                           = "1800s"
  }

  ingress = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  traffic {
    percent         = 100
    type            = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  
}


resource "google_cloud_run_v2_service" "model_training_trigger" {
  name     = "model-training-trigger"
  location = var.region
  project  = var.project_id

  template {
    containers {
      # image = "us-central1-docker.pkg.dev/${var.project_id}/${var.artifact_registry}/model_training_trigger:latest"
      image = local.model_training_trigger_image_uri

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "REGION"
        value = var.region
      }

      env {
        name  = "BUCKET_URI"
        value = var.staging_bucket_uri
      }

      env {
        name  = "IMAGE_URI"
        value = local.model_training_image_uri
      }

      env {
        name  = "MYSQL_HOST"
        value = local.mysql_host
      }

      env {
        name  = "MYSQL_USER"
        value = var.mysql_user
      }

      env {
        name  = "MYSQL_PASSWORD"
        value = var.mysql_password
      }

      env {
        name  = "MYSQL_DATABASE"
        value = var.mysql_database
      }

      env {
        name  = "INSTANCE_CONN_NAME"
        value = local.instance_conn_name
      }

      env {
        name  = "TRAINING_SERVICE_ACCOUNT"
        value = var.service_account_email
      }

      env {
        name  = "VPC_NETWORK"
        value = google_compute_network.airflow_vpc.name   # e.g. "airflow_vpc"
      }

      env {
        name  = "PRIVATE_IP_RANGE_NAME"
        value = google_compute_global_address.private_ip_range.name
      }

      env {
        name  = "PROJECT_NUMBER"
        value = var.project_number
      }


      resources {
        limits = {
          cpu    = "2"    # you can keep this at 1 vCPU
          memory = "8Gi"  # up from the (256 Mi / 512 Mi) default
        }
      }

    }

    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.id
      egress    = "ALL_TRAFFIC"
    }

    service_account = var.service_account_email
    timeout         = "900s"
  }

  # ingress = "INGRESS_TRAFFIC_INTERNAL_ONLY"
  ingress = "INGRESS_TRAFFIC_ALL"

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}


resource "google_cloud_run_v2_service" "backend" {
  name     = "cloudrun-backend"
  location = var.region
  project  = var.project_id

  template {
    containers {
      image = local.backend_image_uri

      env {
        name  = "USE_TCP"
        value = "true"
      }

      env {
        name  = "VM_IP"
        value = local.airflow_lb_ip
      }

      env {
        name  = "MYSQL_HOST"
        value = local.mysql_host
      }

      env {
        name  = "MYSQL_USER"
        value = var.mysql_user
      }

      env {
        name  = "MYSQL_PASSWORD"
        value = var.mysql_password
      }

      env {
        name  = "MYSQL_DATABASE"
        value = var.mysql_database
      }

      env {
        name  = "INSTANCE_CONN_NAME"
        value = local.instance_conn_name
      }


      env {
        name  = "AIRFLOW_ADMIN_USERNAME"
        value = var.airflow_admin_username
      }

      env {
        name  = "AIRFLOW_ADMIN_PASSWORD"
        value = var.airflow_admin_password
      }

      env {
        name  = "AIRFLOW_DAG_ID"
        value = var.airflow_dag_id   # Set this as needed
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "MODEL_SERVING_URL"
        value = google_cloud_run_v2_service.model_serving.uri
      }

      env {
        name  = "GCS_BUCKET_NAME"
        value = var.gcs_bucket_name
      }

      env {
        name  = "VERTEX_REGION"
        value = var.region   # or wherever your Scheduler lives
      }

      env {
        name  = "API_TOKEN"
        value = var.api_token
      }

      # Optional: You can configure more here
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.id
      egress    = "ALL_TRAFFIC"
    }

    # Optional: always allocate CPU
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    service_account       = var.service_account_email
  }

  ingress = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  traffic {
    percent         = 100
    type            = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}


resource "google_cloud_run_v2_service" "model_health_check" {
  name     = "model-health-check"
  location = var.region
  project  = var.project_id

  template {
    containers {
      image = local.model_health_check_image_uri

      env {
        name  = "MYSQL_HOST"
        value = local.mysql_host
      }

      env {
        name  = "MYSQL_USER"
        value = var.mysql_user
      }

      env {
        name  = "MYSQL_PASSWORD"
        value = var.mysql_password
      }

      env {
        name  = "MYSQL_DATABASE"
        value = var.mysql_database
      }

      env {
        name  = "INSTANCE_CONN_NAME"
        value = local.instance_conn_name
      }

      env {
        name  = "GCS_BUCKET_NAME"
        value = var.gcs_bucket_name
      }

      env {
        name  = "VERTEX_REGION"
        value = var.region   # or wherever your Scheduler lives
      }

      env {
        name  = "API_TOKEN"
        value = var.api_token
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "TRAINING_TRIGGER_URL"
        value = "${google_cloud_run_v2_service.model_training_trigger.uri}/trigger-training"
      }

      env {
        name  = "TRAINING_IMAGE_URI"
        value = local.model_training_trigger_image_uri
      }

      

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    # ─── VPC CONNECTOR & EGRESS ──────────────────────────────────
    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.id
      egress    = "ALL_TRAFFIC"
    }

    service_account = var.service_account_email
    timeout         = "300s"
  }

  ingress = "INGRESS_TRAFFIC_INTERNAL_ONLY"

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}


resource "google_compute_region_network_endpoint_group" "cloudrun_neg" {
  name                  = "cloudrun-neg"
  region                = var.region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    # service = var.cloudrun_service_name
    service = google_cloud_run_v2_service.backend.name
  }
}


resource "google_project_iam_member" "cloudsql_client_role" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${var.service_account_email}"
}

resource "google_cloud_run_service_iam_member" "allow_backend" {
  project  = var.project_id
  location = var.region
  service  = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "allow_model_serving" {
  project  = var.project_id
  location = var.region
  service  = google_cloud_run_v2_service.model_serving.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "allow_model_training_trigger" {
  project  = var.project_id
  location = var.region
  service  = google_cloud_run_v2_service.model_training_trigger.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}


resource "google_cloud_run_service_iam_member" "allow_model_health_check" {
  project  = var.project_id
  location = var.region
  service  = google_cloud_run_v2_service.model_health_check.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_pubsub_topic" "retrain_trigger" {
  name    = "model-retrain-trigger"
  project = var.project_id
}


resource "google_pubsub_subscription" "retrain_trigger_push" {
  name  = "retrain-trigger-push"
  topic = google_pubsub_topic.retrain_trigger.name

  # Push directly into your Cloud Run retraining endpoint:
  push_config {
    push_endpoint = "${google_cloud_run_v2_service.model_training_trigger.uri}/trigger-training"
    oidc_token {
      service_account_email = var.service_account_email
      audience              = "${google_cloud_run_v2_service.model_training_trigger.uri}/trigger-training"
    }
  }
}


resource "google_monitoring_notification_channel" "model_retrain_channel" {
  display_name = "Model Retrain Pub/Sub Channel"
  type         = "pubsub"

  labels = {
    topic = google_pubsub_topic.retrain_trigger.id
  }
}

# ─── Custom metric for RMSE ───────────────────────────────────────────────────
resource "google_monitoring_metric_descriptor" "rmse_descriptor" {
  display_name = "Model RMSE"
  type         = "custom.googleapis.com/model/rmse"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  description  = "Root Mean Squared Error of model predictions"
}

# ─── Custom metric for P‑Value ───────────────────────────────────────────────
resource "google_monitoring_metric_descriptor" "p_value_descriptor" {
  display_name = "Model P‑Value"
  type         = "custom.googleapis.com/model/p_value"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"
  description  = "P‑Value from KS test between actuals and predictions"
}

# ─── Alert policy on RMSE > 18 OR P‑Value < 0.05 ────────────────────────────
resource "google_monitoring_alert_policy" "model_retrain_policy" {
  depends_on = [
    google_monitoring_metric_descriptor.rmse_descriptor,
    google_monitoring_metric_descriptor.p_value_descriptor,
  ]

  display_name = "Retrain Model on High Error / Low P‑Value"
  combiner     = "OR"

  conditions {
    display_name = "RMSE above threshold"
    condition_threshold {
      filter          = "resource.type=\"global\" AND metric.type=\"custom.googleapis.com/model/rmse\""
      comparison      = "COMPARISON_GT"
      threshold_value = 18.0
      duration        = "300s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  conditions {
    display_name = "P‑Value below threshold"
    condition_threshold {
      filter          = "resource.type=\"global\" AND metric.type=\"custom.googleapis.com/model/p_value\""
      comparison      = "COMPARISON_LT"
      threshold_value = 0.05
      duration        = "300s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.model_retrain_channel.id
  ]
}