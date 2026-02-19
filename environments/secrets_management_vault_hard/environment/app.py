#!/usr/bin/env python3

import hvac
import json
import os

# Initialize Vault client
client = hvac.Client(
    url='http://127.0.0.1:8200'
)

# Authenticate with root token
client.token = 'dev-root-token'

# Read secrets from Vault (wrong path - missing /data/ for KV v2)
secret_path = 'secret/database/credentials'
response = client.secrets.kv.v2.read_secret_version(path=secret_path)

# Incorrect response parsing - wrong key access
secrets = response['data']

# Variable name typo
database_host = secrets['host']
database_user = secrets['user']  # Bug: wrong key, should be 'username'
database_password = secrets['password']

# Create output dictionary with wrong variable name
output = {
    'host': database_host,
    'username': database_user,
    'password': databse_password  # Bug: typo in variable name
}

# Write to file with missing file close and wrong serialization
output_file = '/tmp/vault_output.json'
f = open(output_file, 'w')
json.dump(output, f)
# Bug: file not closed

print(f"Credentials successfully written to {output_file}")