#!/usr/bin/env python3

import subprocess
import time
import requests
import sys
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

# Global query counter
query_count = 0

# Set up SQLAlchemy event listener to count queries
@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    global query_count
    query_count += 1

# Connect to the database to enable event listening
engine = create_engine('sqlite:///database.db')

# Start the Flask application
print("Starting Flask application...")
app_process = subprocess.Popen(
    ['python', 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for the application to start
print("Waiting for application to start...")
time.sleep(3)

# Reset query counter and make request
query_count = 0
print("\nMaking request to /api/authors endpoint...")

try:
    response = requests.get('http://localhost:5000/api/authors')
    
    print(f"\n{'='*60}")
    print(f"QUERY PERFORMANCE REPORT")
    print(f"{'='*60}")
    print(f"Total SQL queries executed: {query_count}")
    print(f"Response status code: {response.status_code}")
    print(f"\nResponse data:")
    print(response.json())
    print(f"{'='*60}\n")
    
    # Write statistics to file
    with open('/tmp/query_profile.txt', 'w') as f:
        f.write(f"Query Performance Profile\n")
        f.write(f"{'='*60}\n")
        f.write(f"Endpoint: /api/authors\n")
        f.write(f"Total queries executed: {query_count}\n")
        f.write(f"Response status: {response.status_code}\n")
        f.write(f"\nThis profile was generated to identify N+1 query problems.\n")
        f.write(f"With 10 authors in the database:\n")
        f.write(f"  - Expected without optimization: 11 queries (1 + 10)\n")
        f.write(f"  - Expected with optimization: 1-2 queries\n")
        f.write(f"  - Actual: {query_count} queries\n")
    
    print(f"Query statistics saved to /tmp/query_profile.txt")
    
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
    sys.exit(1)
finally:
    # Terminate the Flask application
    print("\nTerminating Flask application...")
    app_process.terminate()
    app_process.wait()

sys.exit(0)