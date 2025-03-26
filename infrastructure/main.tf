# -------------------------------
# 1) Create a VPC
# -------------------------------
resource "google_compute_network" "airflow_vpc" {
  name                    = var.network_name
  project                 = var.project_id
  auto_create_subnetworks = false
}


# Enable the Service Networking API
resource "google_project_service" "servicenetworking" {
  service = "servicenetworking.googleapis.com"
}

# Reserve an internal IP address range for Cloud SQL
resource "google_compute_global_address" "private_ip" {
  name          = "mysql-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.airflow_vpc.self_link
}

# Establish VPC peering between your VPC and Google’s Service Networking
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.airflow_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]
}


# -------------------------------
# 2) Create a Subnet
# -------------------------------
resource "google_compute_subnetwork" "airflow_subnet" {
  name          = var.subnet_name
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.airflow_vpc.self_link
  project       = var.project_id
}

# -------------------------------
# 3) Firewall Rule
# -------------------------------
resource "google_compute_firewall" "airflow_firewall" {
  name    = "allow-airflow-server"
  network = google_compute_network.airflow_vpc.self_link
  allow {
    protocol = "tcp"
    ports    = ["8080", "22"]  # open SSH + Airflow UI
  }
  source_ranges = ["0.0.0.0/0"] # for demo, allow all
}


resource "google_compute_region_autoscaler" "airflow_autoscaler" {
  name    = "airflow-autoscaler"
  project = var.project_id
  region  = var.region

  target = google_compute_region_instance_group_manager.airflow_mig.self_link

    autoscaling_policy {
      cooldown_period = 240
      max_replicas    = 5
      min_replicas    = 1
      mode            = "ON"

      cpu_utilization {
        target            = 0.9
        predictive_method = "NONE"
      }
  }

  depends_on = [google_compute_region_instance_group_manager.airflow_mig]
}