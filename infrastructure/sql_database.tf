#############################################
# 1) Lookup your existing VPC network
#############################################
data "google_compute_network" "existing_vpc" {
  name    = var.network_name  # Replace with your VPC name
  project = var.project_id
}


##############################################################
# Reserve an internal IP range for Cloud SQL private connectivity
##############################################################
resource "google_compute_global_address" "private_ip_address" {
  provider      = google-beta
  name          = var.allocated_ip_range_name  # e.g., "sql-private-ip-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = data.google_compute_network.existing_vpc.self_link
  project       = var.project_id
}

#################################################################
# Create the VPC peering connection with Service Networking
#################################################################
resource "google_service_networking_connection" "private_vpc_connection" {
  provider                = google-beta
  network                 = data.google_compute_network.existing_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}


#########################################################
# Create Cloud SQL Instance with Private Connectivity
#########################################################
resource "google_sql_database_instance" "instance" {
  provider         = google-beta
  name             = "transaction-database"
  region           = var.region
  project          = var.project_id
  database_version = "MYSQL_8_0"

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled      = false
      private_network   = data.google_compute_network.existing_vpc.self_link
      allocated_ip_range = google_compute_global_address.private_ip_address.name
    }
  }
}


# Create a database within the instance
resource "google_sql_database" "database" {
  name     = var.database_name
  instance = google_sql_database_instance.instance.name
  project  = var.project_id
}

# Automatically generate a strong random password
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Automatically generate a random string for the username suffix
resource "random_string" "db_username" {
  length  = 8
  special = false
  upper   = false
}

# Create a dedicated SQL user with generated credentials
resource "google_sql_user" "app_user" {
  name     = "app-${random_string.db_username.result}"
  instance = google_sql_database_instance.instance.name
  password = random_password.db_password.result
  project  = var.project_id
}

resource "null_resource" "create_sales_table" {
  depends_on = [
    google_sql_database_instance.instance,
    google_sql_database.database,
    google_sql_user.app_user
  ]
  
  provisioner "local-exec" {
    command = <<-EOF
      mysql --host=${google_sql_database_instance.instance.ip_address[0].ip_address} \
            --user=${google_sql_user.app_user.name} \
            --password=${random_password.db_password.result} \
            ${google_sql_database.database.name} \
            -e "CREATE TABLE IF NOT EXISTS sales (\`Date\` DATE, \`Product Name\` VARCHAR(255), \`Total Quantity\` INT);"
    EOF
  }
}
