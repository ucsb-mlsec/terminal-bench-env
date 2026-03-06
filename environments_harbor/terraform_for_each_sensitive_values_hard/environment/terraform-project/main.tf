terraform {
  required_version = ">= 1.0"
}

locals {
  db_configs = {
    dev = {
      instance_type = "db.t3.micro"
      storage_gb    = 20
      password      = sensitive("dev-secret-password-123")
    }
    staging = {
      instance_type = "db.t3.small"
      storage_gb    = 50
      password      = sensitive("staging-secret-password-456")
    }
    production = {
      instance_type = "db.t3.large"
      storage_gb    = 100
      password      = sensitive("production-secret-password-789")
    }
  }
}

resource "null_resource" "database_instance" {
  for_each = local.db_configs

  triggers = {
    instance_type = each.value.instance_type
    storage_gb    = each.value.storage_gb
    password_hash = each.value.password
    environment   = each.key
  }

  provisioner "local-exec" {
    command = "echo Creating database instance for ${each.key}"
  }
}

resource "null_resource" "db_credentials" {
  for_each = local.db_configs

  triggers = {
    credential_id = "${each.key}-credentials"
    db_password   = each.value.password
    instance_ref  = each.value.instance_type
  }

  provisioner "local-exec" {
    command = "echo Storing credentials for ${each.key} with password ${each.value.password}"
  }

  depends_on = [null_resource.database_instance]
}

output "database_endpoints" {
  value = {
    for env, config in local.db_configs :
    env => "postgresql://${env}-db.example.com:5432?password=${config.password}"
  }
  description = "Database connection endpoints with embedded credentials"
}

output "database_passwords" {
  value = {
    for env, config in local.db_configs :
    env => config.password
  }
  description = "Database passwords for all environments"
}

output "connection_info" {
  value = [
    for env, config in local.db_configs :
    {
      environment   = env
      instance_type = config.instance_type
      password      = config.password
      connection_string = "host=${env}-db.example.com password=${config.password}"
    }
  ]
  description = "Complete connection information including sensitive data"
}

output "instance_details" {
  value = {
    for env in keys(local.db_configs) :
    env => {
      type     = local.db_configs[env].instance_type
      storage  = local.db_configs[env].storage_gb
      password = local.db_configs[env].password
    }
  }
}