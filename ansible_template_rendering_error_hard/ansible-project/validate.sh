#!/bin/bash

set -e

echo "Starting Ansible template validation..."

# Change to the ansible-project directory
cd /ansible-project

# First, check syntax of the playbook
echo "Checking playbook syntax..."
ansible-playbook site.yml --syntax-check

# Create a temporary inventory file
echo "localhost ansible_connection=local" > /tmp/inventory.tmp

# Run the playbook in check mode to validate template rendering
echo "Validating template rendering..."
ansible-playbook -i /tmp/inventory.tmp site.yml --check

# Clean up
rm -f /tmp/inventory.tmp

echo "✓ All validations passed successfully!"
exit 0