

# -------------------------------
# 1) Create a VPC
# -------------------------------
resource "google_compute_network" "airflow_vpc" {
  name                    = var.network_name
  project                 = var.project_id
  auto_create_subnetworks = false
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





resource "google_compute_network" "my_network" {
  name                    = "my-vpc-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "my_subnet" {
  name          = "my-subnet"
  ip_cidr_range = "10.0.0.0/16"
  network       = google_compute_network.my_network.self_link
  region        = var.region
}


resource "google_compute_global_address" "private_ip_range" {
  name          = "private-ip-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.my_network.self_link
}


resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.my_network.self_link
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]
}


resource "google_compute_firewall" "allow_internal_sql" {
  name    = "allow-internal-sql"
  network = google_compute_network.my_network.name

  allow {
    protocol = "tcp"
    ports    = ["3306"]
  }

  # Allow all resources in your VPC (adjust source_ranges as needed)
  source_ranges = ["10.0.0.0/16"]
}


resource "google_project_service" "vpc_access_api" {
  project = var.project_id
  service = "vpcaccess.googleapis.com"

  disable_on_destroy = false
}


resource "google_vpc_access_connector" "cloudrun_connector" {
  name          = "cloudrun-connector"
  region        = var.region
  network       = google_compute_network.airflow_vpc.self_link
  ip_cidr_range = "10.8.0.0/28"

  depends_on = [
    google_project_service.vpc_access_api
  ]
}


resource "google_compute_firewall" "allow_cloudrun_to_airflow" {
  name    = "allow-cloudrun-to-airflow"
  network = google_compute_network.airflow_vpc.self_link

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["10.8.0.0/28"]
  target_tags   = ["airflow-server"]
}