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


# Create a database within the instance
resource "google_sql_database" "database" {
  name     = var.database_name
  instance = google_sql_database_instance.instance.name
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
}

# Use a null_resource to run a SQL command that creates the 'sales' table
resource "null_resource" "create_sales_table" {
  depends_on = [
    google_sql_database_instance.mysql_instance,
    google_sql_database.database,
    google_sql_user.app_user
  ]

  provisioner "local-exec" {
    command = <<-EOF
      mysql --host=${google_sql_database_instance.mysql_instance.ip_address[0].ip_address} \
            --user=${google_sql_user.app_user.name} \
            --password=${random_password.db_password.result} \
            ${google_sql_database.database.name} \
            -e "CREATE TABLE IF NOT EXISTS sales (\`Date\` DATE, \`Product Name\` VARCHAR(255), \`Total Quantity\` INT);"
    EOF
  }
}
