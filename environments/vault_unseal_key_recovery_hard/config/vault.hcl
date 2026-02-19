# HashiCorp Vault Production Configuration
# Instance: vault-prod-01.example.com
# Initialized: 2024-01-15
# Shamir Secret Sharing Configuration:
#   - Total Key Shares: 5
#   - Threshold Required: 3
#   - Key shares were distributed to authorized operators
#   - This configuration does NOT store unseal keys

# Storage backend configuration
storage "file" {
  path = "/opt/vault/data"
}

# TCP listener configuration
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_disable   = false
  tls_cert_file = "/etc/vault.d/tls/cert.pem"
  tls_key_file  = "/etc/vault.d/tls/key.pem"
  tls_min_version = "tls12"
}

# API address for client connections
api_addr = "https://vault-prod-01.example.com:8200"

# Cluster address for replication and HA
cluster_addr = "https://vault-prod-01.example.com:8201"

# Enable the Vault UI
ui = true

# Logging configuration
log_level = "info"
log_format = "json"

# Seal configuration - Shamir's Secret Sharing
seal "shamir" {
  # This vault uses Shamir's Secret Sharing
  # No additional configuration required
  # Unseal keys are NOT stored in this configuration
}

# Disable memory locking for security
disable_mlock = false

# Default lease TTL settings
default_lease_ttl = "768h"  # 32 days
max_lease_ttl     = "8760h" # 365 days

# Telemetry configuration
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname          = false
}

# Performance tuning
max_lease_ttl_seconds = 31536000