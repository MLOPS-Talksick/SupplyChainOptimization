resource "google_sql_database_instance" "instance" {
  provider         = google-beta
  name             = "transaction-database"
  region           = var.region
  database_version = "MYSQL_8_0"

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "allowed-network"
        value = "0.0.0.0/0"  # e.g., "203.0.113.45/32" or "0.0.0.0/0" (for demo only)
      }
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
