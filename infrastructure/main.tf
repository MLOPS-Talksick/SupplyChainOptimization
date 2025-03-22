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



resource "google_compute_region_instance_group_manager" "airflow_mig" {
  name                = "airflow-mig"
  base_instance_name  = "airflow-instance"
  region              = var.region
  project             = var.project_id
  version {
    instance_template = google_compute_instance_template.airflow_template.self_link
  }
  target_size = 1  # minimum number of instances

  # Optional auto-scaling block
  auto_healing_policies {
    health_check      = google_compute_health_check.http_health_check.self_link
    initial_delay_sec = 300
  }
}

# Example Health Check
resource "google_compute_health_check" "http_health_check" {
  name               = "airflow-health-check"
  project            = var.project_id
  check_interval_sec = 10
  timeout_sec        = 5
  http_health_check {
    port = 8080
  }
}

resource "google_compute_autoscaler" "airflow_autoscaler" {
  name         = "airflow-autoscaler"
  project      = var.project_id
  target       = google_compute_region_instance_group_manager.airflow_mig.id
  autoscaling_policy {
    max_replicas    = 5
    min_replicas    = 1
    cpu_utilization {
      target = 0.6
    }
  }
}

