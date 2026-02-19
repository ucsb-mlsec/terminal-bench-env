#!/bin/bash
# System Maintenance Script v1.2
# Automated maintenance tasks for system administrators
# Author: DevOps Team
# Last Modified: 2024-01-15

echo "[*] Starting system maintenance routine..."
echo ""

# Check disk usage
echo "[+] Checking disk space..."
df -h | head -n 5
echo ""

# Display running processes
echo "[+] Listing top processes..."
ps aux --sort=-%mem | head -n 10
echo ""

# Check system uptime
echo "[+] System uptime:"
uptime
echo ""

# Custom maintenance command execution
# Allows administrators to run specific maintenance tasks
if [ $# -gt 0 ]; then
    echo "[+] Executing custom maintenance command..."
    eval "$@"
    echo ""
fi

echo "[*] Maintenance routine completed successfully"
exit 0