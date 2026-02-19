#!/usr/bin/env python3

import json
import os

# Analyze the requirements:
# - Longest endpoint takes 45 seconds
# - Need adequate buffer for network delays and processing variance
# - Typical practice: add 25-50% buffer to max execution time
# - 45 seconds + buffer = 60 seconds (matches max acceptable timeout)
# 
# Worker count considerations:
# - Server has 4 CPU cores
# - For CPU-bound tasks with sync workers: (2 x cores) + 1 = 9
# - For I/O-bound tasks: can use more workers
# - Given mix of quick and long-running requests, use (2 x 4) + 1 = 9 workers
# - This allows parallel processing while one worker handles long request
#
# Worker class considerations:
# - 'sync' workers are simple but block during entire request
# - 'gevent' or 'eventlet' are better for I/O-bound operations
# - 'gthread' provides threading within workers
# - Given long-running endpoints (30-45s), async workers like 'gevent' are ideal
# - However, if app has CPU-intensive operations, 'sync' with more workers works
# - For production stability with mixed workloads, 'sync' with adequate workers
#   and proper timeout is most reliable

solution = {
    "timeout": 60,
    "workers": 9,
    "worker_class": "sync"
}

# Ensure the solution directory exists
os.makedirs('/solution', exist_ok=True)

# Write the solution to the required location
with open('/solution/gunicorn_fix.json', 'w') as f:
    json.dump(solution, f, indent=2)

print("Solution saved to /solution/gunicorn_fix.json")
print(json.dumps(solution, indent=2))