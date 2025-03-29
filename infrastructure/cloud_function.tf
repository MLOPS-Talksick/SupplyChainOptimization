# 1. Upload your Cloud Function source code (ZIP file) to the bucket.
resource "google_storage_bucket_object" "cloud_function_zip" {
  name   = "cloud_function_source.zip"                                 # Object name in the bucket
  bucket = google_storage_bucket.buckets["fully-processed-data-test"].name # Reference to your bucket using for_each key
  source = "${path.module}/Cloudrun_Function/GCS_TO_SQL.zip"                  # Local path to your ZIP file

  depends_on = [
    google_storage_bucket.buckets["fully-processed-data-test"]
  ]
}

# 2. Deploy the Cloud Function with an event trigger on the bucket.
resource "google_cloudfunctions_function" "process_data_function" {
  name                  = "processDataFunction"
  description           = "Triggered when a file is ingested into fully-processed-data-test bucket"
  runtime               = "python39"                                  # Update runtime if needed
  available_memory_mb   = 256                                         # Adjust memory if needed
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