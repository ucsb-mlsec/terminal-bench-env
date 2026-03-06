#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template_string
import os
import subprocess
import re

app = Flask(__name__)

@app.route('/')
def index():
    """
    Home page - safe route
    """
    return "Welcome to the Web Application"

@app.route('/health')
def health():
    """
    Health check endpoint - safe route
    Returns application health status
    """
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/api/ping', methods=['GET', 'POST'])
def ping_host():
    """
    VULNERABLE: Ping a host provided by user
    Takes 'host' parameter and pings it
    """
    if request.method == 'POST':
        host = request.json.get('host', '')
    else:
        host = request.args.get('host', '')
    
    if host:
        # VULNERABLE: Direct command injection possible
        command = f"ping -c 4 {host}"
        result = os.system(command)
        return jsonify({"command": command, "result": result})
    
    return jsonify({"error": "No host provided"}), 400

@app.route('/api/users')
def get_users():
    """
    Safe route - returns mock user data
    No system commands executed
    """
    users = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ]
    return jsonify(users)

@app.route('/admin/backup', methods=['POST'])
def backup_data():
    """
    VULNERABLE: Backup data to user-specified path
    Takes 'path' parameter from form data
    """
    backup_path = request.form.get('path', '/tmp/backup')
    
    # VULNERABLE: User input directly in command
    command = f"tar -czf {backup_path}/backup.tar.gz /var/data"
    subprocess.call(command, shell=True)
    
    return jsonify({"message": "Backup completed", "path": backup_path})

@app.route('/user/export', methods=['GET'])
def export_user_data():
    """
    VULNERABLE: Export user data with custom filename
    Takes 'filename' parameter from query string
    """
    filename = request.args.get('filename', 'export.csv')
    
    # VULNERABLE: Filename not sanitized
    command = f"echo 'user,email,age' > /tmp/{filename} && cat /tmp/{filename}"
    result = os.popen(command).read()
    
    return jsonify({"output": result, "filename": filename})

@app.route('/system/check', methods=['POST'])
def check_service():
    """
    VULNERABLE: Check if a service is running
    Takes 'service' parameter from JSON body
    """
    service_name = request.json.get('service', '')
    
    if service_name:
        # VULNERABLE: Direct injection into systemctl command
        cmd = f"systemctl status {service_name}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        return jsonify({
            "service": service_name,
            "output": stdout.decode(),
            "error": stderr.decode()
        })
    
    return jsonify({"error": "No service specified"}), 400

@app.route('/tools/convert', methods=['POST'])
def convert_file():
    """
    VULNERABLE: Convert file format using user-provided parameters
    Takes 'input_file' and 'output_format' from form data
    """
    input_file = request.form.get('input_file', '')
    output_format = request.form.get('output_format', 'pdf')
    
    if input_file:
        # VULNERABLE: User input in command without sanitization
        command = f"convert {input_file} output.{output_format}"
        subprocess.run(command, shell=True)
        return jsonify({"message": "Conversion started", "input": input_file, "format": output_format})
    
    return jsonify({"error": "No input file provided"}), 400

@app.route('/static/page')
def static_page():
    """
    Safe route - renders a static template
    No system commands or user input processing
    """
    template = """
    <html>
        <head><title>Static Page</title></head>
        <body>
            <h1>Welcome to our static page</h1>
            <p>This is a safe endpoint with no user input processing.</p>
        </body>
    </html>
    """
    return render_template_string(template)

@app.route('/api/search', methods=['GET'])
def search_data():
    """
    Safe route - searches mock database
    Uses parameterized queries (simulated)
    """
    query = request.args.get('q', '')
    
    # Safe: No system commands, just string matching
    mock_data = [
        "Python programming",
        "Flask framework",
        "Web development",
        "Security best practices",
        "Database management"
    ]
    
    results = [item for item in mock_data if query.lower() in item.lower()]
    return jsonify({"query": query, "results": results})

@app.route('/logs/view', methods=['GET'])
def view_logs():
    """
    VULNERABLE: View log files with user-specified log name
    Takes 'logfile' parameter from query string
    """
    logfile = request.args.get('logfile', 'app.log')
    
    # VULNERABLE: Log filename directly in command
    command = f"tail -n 100 /var/log/{logfile}"
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    
    return jsonify({"logfile": logfile, "content": output.decode()})

@app.route('/api/status')
def api_status():
    """
    Safe route - returns API status information
    No user input or system commands
    """
    return jsonify({
        "api": "operational",
        "endpoints": 11,
        "uptime": "99.9%"
    })

@app.route('/admin/cleanup', methods=['POST'])
def cleanup_temp():
    """
    Safe route - cleans temporary files with hardcoded paths
    No user input used in commands
    """
    # Safe: No user input, hardcoded safe path
    safe_path = "/tmp/app_temp"
    if os.path.exists(safe_path):
        subprocess.run(["rm", "-rf", safe_path])
    
    return jsonify({"message": "Cleanup completed", "path": safe_path})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

Now let me analyze the code and identify the vulnerable routes:

```python
#!/usr/bin/env python3

# Analysis script to identify vulnerable routes
vulnerable_routes = []

# /api/ping - VULNERABLE: os.system() with user input from request.args/request.json
vulnerable_routes.append('/api/ping')

# /admin/backup - VULNERABLE: subprocess.call() with shell=True and user input from request.form
vulnerable_routes.append('/admin/backup')

# /user/export - VULNERABLE: os.popen() with user input from request.args
vulnerable_routes.append('/user/export')

# /system/check - VULNERABLE: subprocess.Popen() with shell=True and user input from request.json
vulnerable_routes.append('/system/check')

# /tools/convert - VULNERABLE: subprocess.run() with shell=True and user input from request.form
vulnerable_routes.append('/tools/convert')

# /logs/view - VULNERABLE: subprocess.check_output() with shell=True and user input from request.args
vulnerable_routes.append('/logs/view')

# Write to output file
with open('/tmp/vulnerable_routes.txt', 'w') as f:
    for route in vulnerable_routes:
        f.write(route + '\n')