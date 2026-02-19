output "all_database_ids" {
  description = "Map of environment names to database instance IDs"
  value = {
    for env, instance in null_resource.database_instance : env => instance.id
  }
}

output "database_connection_strings" {
  description = "Connection strings for all database instances"
  value = {
    for env, config in local.db_configs : env => "postgresql://admin:${config.password}@${config.host}/dbname"
  }
}

output "environment_summary" {
  description = "Summary of all database configurations"
  value = {
    for env, config in local.db_configs : env => {
      instance_type = config.instance_type
      password      = config.password
      host          = config.host
      environment   = env
    }
  }
}

output "monitoring_enabled" {
  description = "Whether monitoring is enabled for database instances"
  value       = var.enable_monitoring
}