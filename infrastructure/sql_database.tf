# Create a MySQL Cloud SQL instance with private IP (or adjust for authorized networks)
resource "google_sql_database_instance" "mysql_instance" {
  name             = "transaction-database"
  database_version = "MYSQL_8_0"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    # Option 1: Using Private IP (if your VM is in the same VPC)
    ip_configuration {
      ipv4_enabled   = false
      private_network = google_compute_network.airflow_vpc.self_link
    }
  }
}

# Create a database within the instance
resource "google_sql_database" "database" {
  name     = "transactions"
  instance = google_sql_database_instance.mysql_instance.name
}

# Create a dedicated SQL user for your application
resource "google_sql_user" "app_user" {
  name     = var.db_username
  instance = google_sql_database_instance.mysql_instance.name
  password = var.db_password
}

# Use a null_resource to run a SQL command and create your table
resource "null_resource" "create_sales_table_mysql" {
  depends_on = [
    google_sql_database_instance.mysql_instance,
    google_sql_database.database,
    google_sql_user.app_user
  ]

  provisioner "local-exec" {
    command = <<-EOF
      mysql --host=${google_sql_database_instance.mysql_instance.ip_address[0].ip_address} \
            --user=${var.db_username} \
            --password=${var.db_password} \
            ${google_sql_database.database.name} \
            -e "CREATE TABLE IF NOT EXISTS sales (\`Date\` DATE, \`Product Name\` VARCHAR(255), \`Total Quantity\` INT);"
    EOF
  }
}
