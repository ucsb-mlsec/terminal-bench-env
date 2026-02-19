#!/usr/bin/env python3

import os

# Create the application directory structure
os.makedirs('/opt/celery_app/data', exist_ok=True)

# Create celeryconfig.py with problematic timeout
celeryconfig_content = '''# Celery Configuration File
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# Task serialization settings
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

# Timeout settings - currently too aggressive
task_soft_time_limit = 30  # Soft timeout in seconds
task_time_limit = 60  # Hard timeout in seconds

# Other settings
task_track_started = True
task_acks_late = True
worker_prefetch_multiplier = 1
'''

with open('/opt/celery_app/celeryconfig.py', 'w') as f:
    f.write(celeryconfig_content)

# Create tasks.py
tasks_content = '''from celery import Celery
import time
import os

app = Celery('tasks')
app.config_from_object('celeryconfig')

@app.task(name='tasks.process_file')
def process_file(filepath):
    """Process a data file - simulates CPU-intensive work"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Simulate processing based on file size
    file_size = os.path.getsize(filepath)
    processing_time = file_size / 1000  # 1 second per KB
    
    start_time = time.time()
    time.sleep(processing_time)
    elapsed = time.time() - start_time
    
    return {
        'file': filepath,
        'size_bytes': file_size,
        'processing_time': elapsed,
        'status': 'completed'
    }
'''

with open('/opt/celery_app/tasks.py', 'w') as f:
    f.write(tasks_content)

# Create process_data.py
process_data_content = '''#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/opt/celery_app')

from tasks import process_file

def main():
    data_dir = '/opt/celery_app/data'
    files = ['small.dat', 'medium.dat', 'large.dat']
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Processing {filename}...")
            try:
                result = process_file.delay(filepath)
                output = result.get(timeout=120)
                print(f"  Success: {output}")
            except Exception as e:
                print(f"  Failed: {e}")
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main()
'''

with open('/opt/celery_app/process_data.py', 'w') as f:
    f.write(process_data_content)
os.chmod('/opt/celery_app/process_data.py', 0o755)

# Create test data files
# small.dat - processes in ~5 seconds (5KB)
with open('/opt/celery_app/data/small.dat', 'wb') as f:
    f.write(b'X' * 5000)

# medium.dat - processes in ~45 seconds (45KB)
with open('/opt/celery_app/data/medium.dat', 'wb') as f:
    f.write(b'X' * 45000)

# large.dat - processes in ~90 seconds (90KB)
with open('/opt/celery_app/data/large.dat', 'wb') as f:
    f.write(b'X' * 90000)

# Now solve the problem by analyzing the configuration
# The issue: task_time_limit is 60 seconds, but large.dat needs 90 seconds
# Solution: Set timeout to at least 90 seconds, but keep it reasonable (under 300)
# Recommended value: 120 seconds (gives buffer for large.dat and future growth)

solution_content = '''timeout_seconds=120
config_file=/opt/celery_app/celeryconfig.py
'''

with open('/opt/celery_app/solution.txt', 'w') as f:
    f.write(solution_content)

print("Celery application structure created successfully!")
print("\nCurrent situation:")
print("- celeryconfig.py has task_time_limit=60 seconds")
print("- small.dat processes in ~5 seconds ✓")
print("- medium.dat processes in ~45 seconds ✗ (exceeds soft limit)")
print("- large.dat processes in ~90 seconds ✗ (exceeds hard limit)")
print("\nSolution saved to /opt/celery_app/solution.txt")
print("Recommended timeout: 120 seconds")