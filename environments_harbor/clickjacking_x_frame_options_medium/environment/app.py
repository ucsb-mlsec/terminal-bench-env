#!/usr/bin/env python3
# Script to add clickjacking protection to Flask app

# Read the original app.py
with open('app.py', 'r') as f:
    content = f.read()

# Apply modifications to add X-Frame-Options header
# We need to add an after_request handler that sets the X-Frame-Options header

# Find the position after the app = Flask(__name__) line to insert the after_request decorator
lines = content.split('\n')
modified_lines = []

app_created = False
after_request_added = False

for i, line in enumerate(lines):
    modified_lines.append(line)
    
    # After we find "app = Flask(__name__)", add the after_request handler
    if 'app = Flask(__name__)' in line and not after_request_added:
        app_created = True
        # Add blank line and the after_request decorator function
        modified_lines.append('')
        modified_lines.append('@app.after_request')
        modified_lines.append('def add_security_headers(response):')
        modified_lines.append('    response.headers[\'X-Frame-Options\'] = \'DENY\'')
        modified_lines.append('    return response')
        after_request_added = True

modified_content = '\n'.join(modified_lines)

# Write back the modified content
with open('app.py', 'w') as f:
    f.write(modified_content)

print("Clickjacking protection applied successfully")