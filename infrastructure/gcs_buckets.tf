resource "google_storage_bucket" "buckets" {
  for_each = var.bucket_names
  name     = each.value
  location = "US"
  project  = var.project_id

  # Set force_destroy if needed:
  force_destroy = false

  versioning {
    enabled = true
  }

}