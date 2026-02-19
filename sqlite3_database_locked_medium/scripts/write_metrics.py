#!/usr/bin/env python3

import sqlite3
import sys
import random
import time
from datetime import datetime

# Connect to database with default settings (no timeout, no WAL mode)
conn = sqlite3.connect('/workspace/data/metrics.db')
cursor = conn.cursor()

# Get service name from command line or use default
if len(sys.argv) > 1:
    service_name = sys.argv[1]
else:
    service_name = f"service_{random.randint(1, 5)}"

# Generate random metrics
response_time_ms = random.randint(50, 500)
status_code = random.choice([200, 201, 400, 500])
timestamp = datetime.now().isoformat()

# Insert data into database
cursor.execute(
    "INSERT INTO performance_metrics (timestamp, service_name, response_time_ms, status_code) VALUES (?, ?, ?, ?)",
    (timestamp, service_name, response_time_ms, status_code)
)

# Commit and close
conn.commit()
conn.close()

print(f"Inserted metric: {service_name} - {response_time_ms}ms - {status_code}")