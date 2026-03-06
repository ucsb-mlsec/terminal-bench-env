#!/usr/bin/env python3

from flask import Flask, request, jsonify
import subprocess
import json
import re

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "mock-elasticsearch",
        "version": "1.4.0",
        "tagline": "You Know, for Search"
    })

@app.route('/_search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Invalid query"}), 400
        
        query = data['query']
        script = None
        
        # Extract script from various possible structures
        if isinstance(query, dict):
            if 'script' in query:
                script = query['script']
            elif 'script_fields' in query:
                script = query['script_fields']
            elif 'filtered' in query and 'query' in query['filtered']:
                if 'script' in query['filtered']['query']:
                    script = query['filtered']['query']['script']
        
        if script is None:
            return jsonify({"hits": {"total": 0, "hits": []}})
        
        # Execute the script
        result = execute_groovy_script(script)
        
        return jsonify({
            "hits": {
                "total": 1,
                "hits": [{
                    "_source": {
                        "result": result
                    }
                }]
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def execute_groovy_script(script):
    """
    Simulate Groovy script execution by parsing common patterns
    and executing system commands
    """
    
    # Pattern 1: 'command'.execute().text
    pattern1 = r"'([^']+)'\.execute\(\)(?:\.text)?"
    match = re.search(pattern1, script)
    if match:
        command = match.group(1)
        return execute_command(command)
    
    # Pattern 2: "command".execute().text
    pattern2 = r'"([^"]+)"\.execute\(\)(?:\.text)?'
    match = re.search(pattern2, script)
    if match:
        command = match.group(1)
        return execute_command(command)
    
    # Pattern 3: Runtime.getRuntime().exec("command")
    pattern3 = r'Runtime\.getRuntime\(\)\.exec\(["\']([^"\']+)["\']\)'
    match = re.search(pattern3, script)
    if match:
        command = match.group(1)
        return execute_command(command)
    
    # Pattern 4: java.lang.Runtime.getRuntime().exec("command")
    pattern4 = r'java\.lang\.Runtime\.getRuntime\(\)\.exec\(["\']([^"\']+)["\']\)'
    match = re.search(pattern4, script)
    if match:
        command = match.group(1)
        return execute_command(command)
    
    # Pattern 5: Process proc = "command".execute()
    pattern5 = r'(?:def|Process)\s+\w+\s*=\s*["\']([^"\']+)["\']\.execute\(\)'
    match = re.search(pattern5, script)
    if match:
        command = match.group(1)
        return execute_command(command)
    
    return "Script executed but no recognizable command pattern found"

def execute_command(command):
    """
    Execute a system command and return its output
    """
    try:
        result = subprocess.check_output(
            command,
            shell=True,
            stderr=subprocess.STDOUT,
            timeout=5
        )
        return result.decode('utf-8').strip()
    except subprocess.TimeoutExpired:
        return "Command execution timed out"
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.output.decode('utf-8').strip()}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)