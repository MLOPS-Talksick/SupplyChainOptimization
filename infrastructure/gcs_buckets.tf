resource "google_storage_bucket" "full_raw_data_test" {
  name     = "full-raw-data-test"
  location = "US"  # "US" is the multi-region location
  project  = var.project_id

#   # If you want to allow Terraform to destroy non-empty buckets automatically:
#   force_destroy = true

  versioning {
    enabled = true
  }

}

resource "google_storage_bucket" "fully_processed_data_test" {
  name     = "fully-processed-data-test"
  location = "US"
  project  = var.project_id
#   force_destroy = true

  versioning {
    enabled = true
  }
}