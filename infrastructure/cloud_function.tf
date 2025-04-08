# 1. Upload your Cloud Function source code (ZIP file) to the bucket.
resource "google_storage_bucket_object" "cloud_function_zip" {
  name   = "cloud_function_source.zip"                                 # Object name in the bucket
  bucket = google_storage_bucket.buckets["fully-processed-data-test"].name # Reference to your bucket using for_each key
  source = "${path.module}/../Cloudrun_Function/GCS_TO_SQL.zip"

  depends_on = [
    google_storage_bucket.buckets["fully-processed-data-test"]
  ]
}

# 2. Deploy the Cloud Function with an event trigger on the bucket.
resource "google_cloudfunctions_function" "process_data_function" {
  name                  = "processDataFunction"
  description           = "Triggered when a file is ingested into fully-processed-data-test bucket"
  runtime               = "python39"                                  # Update runtime if needed
  available_memory_mb   = 512                                         # Adjust memory if needed
  source_archive_bucket = google_storage_bucket.buckets["fully-processed-data-test"].name
  source_archive_object = google_storage_bucket_object.cloud_function_zip.name
  entry_point           = "main"                                      # Update with your function's entry point

  # Set up the trigger so the function is invoked when an object is finalized in the bucket.
  event_trigger {
    event_type = "google.storage.object.finalize"
    resource   = google_storage_bucket.buckets["fully-processed-data-test"].name
  }

  depends_on = [
    google_storage_bucket.buckets["fully-processed-data-test"],
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
        value = var.mysql_host
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
    }

    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.name
      egress    = "ALL_TRAFFIC"
    }

    service_account = var.service_account_email

    max_instance_request_concurrency = 80  # optional: adjust if needed
    timeout                           = "900s"
  }

  ingress = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

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
        value = local.model_training_trigger_image_uri
      }

      env {
        name  = "IMAGE_TAG_HASH"
        value = local.model_training_trigger_image_uri
      }
    }

    vpc_access {
      connector = google_vpc_access_connector.cloudrun_connector.name
      egress    = "ALL_TRAFFIC"
    }

    service_account = var.service_account_email
    timeout         = "900s"
  }

  ingress = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}


resource "google_cloud_run_v2_job" "model_training_job" {
  name     = "model-training-job"
  location = var.region
  project  = var.project_id

  template {
    template {
      containers {
        # image = "us-central1-docker.pkg.dev/${var.project_id}/${var.artifact_registry}/model_training:latest"
        image = local.model_training_image_uri

        env {
          name  = "MYSQL_HOST"
          value = var.mysql_host
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
          value = local.model_training_image_uri
        }
      }

      max_retries    = 1
      timeout        = "900s"  # Adjust based on training duration
      service_account = var.service_account_email
    }
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
        name  = "VM_IP"
        value = local.airflow_lb_ip
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
      connector = google_vpc_access_connector.cloudrun_connector.name
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

  depends_on = [
    google_compute_global_forwarding_rule.airflow_http_forwarding_rule
  ]
}