#!/bin/bash

echo "Stopping server..."
pkill -f "python3 /opt/server/server.py"
killall -9 python3 2>/dev/null
sleep 1

echo "Starting server..."
cd /opt/server
python3 /opt/server/server.py > /var/log/https_server.log 2>&1 &

sleep 2
echo "Server restarted on port 8443"