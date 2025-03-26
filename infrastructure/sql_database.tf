resource "google_compute_network" "private_network" {
  provider = google-beta
  name     = "private-network"
}

resource "google_compute_global_address" "private_ip_address" {
  provider     = google-beta
  name         = "private-ip-address"
  purpose      = "VPC_PEERING"
  address_type = "INTERNAL"
  prefix_length = 16
  network      = google_compute_network.private_network.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  provider = google-beta
  network  = google_compute_network.private_network.id
  service  = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}


resource "google_sql_database_instance" "instance" {
  provider         = google-beta
  name             = "mysql-instance"
  database_version = "MYSQL_8_0"
  region           = var.region

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    tier = "db-f1-micro"

    ip_configuration {
      ipv4_enabled = false
      private_network = google_compute_network.private_network.self_link
      
      # This line is required to assign an IP from your reserved range:
      allocated_ip_range = google_compute_global_address.private_ip_address.name
      
      enable_private_path_for_google_cloud_services = true
    }
  }
}
