#!/usr/bin/env python3

from flask import Flask, jsonify
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that returns immediately"""
    try:
        logger.info("Health check requested")
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/quick', methods=['GET'])
def quick():
    """Quick data query endpoint"""
    try:
        logger.info("Quick query requested")
        time.sleep(0.5)
        logger.info("Quick query completed")
        return jsonify({"data": "quick response"}), 200
    except Exception as e:
        logger.error(f"Quick query failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/report', methods=['GET'])
def report():
    """Report generation endpoint - takes 30 seconds"""
    try:
        logger.info("Report generation requested")
        time.sleep(30)
        logger.info("Report generation completed")
        return jsonify({"report": "generated"}), 200
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/background', methods=['GET'])
def background():
    """Background processing endpoint - takes 45 seconds"""
    try:
        logger.info("Background processing requested")
        time.sleep(45)
        logger.info("Background processing completed")
        return jsonify({"result": "processed"}), 200
    except Exception as e:
        logger.error(f"Background processing failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

Now, for the Gunicorn configuration fix, I need to create the JSON solution file:

```python
#!/usr/bin/env python3

import json
import os

# Analysis of the requirements:
# - Longest endpoint takes 45 seconds
# - Maximum acceptable timeout is 60 seconds
# - Need adequate buffer beyond 45 seconds
# - Server has 4 CPU cores
# - Need to handle varying execution times efficiently

# Solution:
# 1. Timeout: 60 seconds (allows 45-second endpoint to complete with 15-second buffer)
# 2. Workers: 9 (formula: 2 * CPU_cores + 1 = 2 * 4 + 1 = 9, production best practice)
# 3. Worker class: "sync" (simple and reliable for blocking operations like time.sleep())
#    Alternative consideration: "gevent" or "gthread" for better concurrency,
#    but "sync" is most straightforward for this use case with multiple workers

solution = {
    "timeout": 60,
    "workers": 9,
    "worker_class": "sync"
}

# Create the solution directory if it doesn't exist
os.makedirs('/solution', exist_ok=True)

# Write the solution to the file
with open('/solution/gunicorn_fix.json', 'w') as f:
    json.dump(solution, f, indent=2)

print("Solution saved to /solution/gunicorn_fix.json")
print(json.dumps(solution, indent=2))