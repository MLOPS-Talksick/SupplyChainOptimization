# Create a MySQL Cloud SQL instance using private IP
resource "google_sql_database_instance" "mysql_instance" {
  name             = "mysql-instance"
  database_version = "MYSQL_8_0"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    ip_configuration {
      ipv4_enabled      = false
      private_network   = google_compute_network.airflow_vpc.self_link
      allocated_ip_range = google_compute_global_address.private_ip.name
    }
  }
}

# Create a database within the instance
resource "google_sql_database" "database" {
  name     = var.database_name
  instance = google_sql_database_instance.mysql_instance.name
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

# Create a dedicated SQL user with an auto-generated username and password
resource "google_sql_user" "app_user" {
  name     = "app-${random_string.db_username.result}"
  instance = google_sql_database_instance.mysql_instance.name
  password = random_password.db_password.result
}

# Use a null_resource with a local-exec provisioner to create your table
resource "null_resource" "create_sales_table" {
  depends_on = [
    google_sql_database_instance.mysql_instance,
    google_sql_database.database,
    google_sql_user.app_user
  ]
  provisioner "local-exec" {
    command = <<-EOF
      mysql --host=$(gcloud sql instances describe ${google_sql_database_instance.mysql_instance.name} --format="value(ipAddresses.ipAddress)" --quiet) \
            --user=${google_sql_user.app_user.name} \
            --password=${random_password.db_password.result} \
            ${google_sql_database.database.name} \
            -e "CREATE TABLE IF NOT EXISTS sales (\`Date\` DATE, \`Product Name\` VARCHAR(255), \`Total Quantity\` INT);"
    EOF
  }
}
