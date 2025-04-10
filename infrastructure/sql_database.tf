# Create the Cloud SQL instance
resource "google_sql_database_instance" "instance" {
  provider         = google-beta
  name             = "mlops-sql-2"
  region           = var.region
  database_version = "MYSQL_8_0"

  settings {
    tier = "db-f1-micro"

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.my_network.self_link
    }
  }

  lifecycle {
    ignore_changes = [deletion_protection]
  }

  depends_on = [
    google_service_networking_connection.private_vpc_connection,
    google_compute_global_address.private_ip_range,
    google_compute_subnetwork.my_subnet,
  ]
}

# Create the database in the instance
resource "google_sql_database" "database" {
  name     = var.mysql_database
  instance = google_sql_database_instance.instance.name
}

# Create the SQL user
resource "google_sql_user" "user" {
  name     = var.mysql_user
  instance = google_sql_database_instance.instance.name
  password = var.mysql_password
}