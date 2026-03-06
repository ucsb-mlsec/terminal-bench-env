#!/bin/bash

# Locust Worker Startup Script
# This script starts multiple Locust worker nodes to connect to the master

# Configuration
MASTER_HOST="localhost"
MASTER_PORT=5557
LOCUSTFILE="locustfile.py"

# Starting worker 1
echo "Starting Locust worker 1..."
locust --worker --master-host=$MASTER_HOST --master-port=$MASTER_PORT -f $LOCUSTFILE &

# Starting worker 2
echo "Starting Locust worker 2..."
locust --worker --master-host=127.0.0.1 --master-port=5557 -f locustfile.py &

# Starting worker 3
echo "Starting Locust worker 3..."
locust --worker --master-host=$MASTER_HOST --master-port=$MASTER_PORT -f $LOCUSTFILE &

# Starting worker 4
echo "Starting Locust worker 4..."
locust --worker --master-host=127.0.0.1 --master-port=5557 -f locustfile.py &

echo "All worker nodes started successfully"
echo "Workers connecting to master at $MASTER_HOST:$MASTER_PORT"