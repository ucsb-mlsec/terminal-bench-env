#!/bin/bash

# Start Locust Master Node
# Production Load Test Configuration

cd /opt/locust_swarm

locust -f locustfile.py \
  --master \
  --host http://api.example.local:8080 \
  --users 1000 \
  --spawn-rate 50 \
  --run-time 300 \
  --master-bind-port 5557