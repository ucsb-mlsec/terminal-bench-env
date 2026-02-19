#!/bin/bash

cd /opt/webapp

echo "Starting web application..."

python3 app.py &

sleep 2

echo "Web application started on http://localhost:8080"