#!/usr/bin/env python3
# Token Validator - Do Not Modify

import sys
import base64
import subprocess
import os
from datetime import datetime

# Compile the proto file
proto_dir = '/challenge'
os.chdir(proto_dir)
subprocess.run(['protoc', '--python_out=.', 'schema.proto'], check=True)

# Import the generated protobuf module
import auth_token_pb2

def validate_token(token_path):
    try:
        # Read the token file
        with open(token_path, 'r') as f:
            token_b64 = f.read().strip()
        
        # Decode base64
        token_bytes = base64.b64decode(token_b64)
        
        # Parse protobuf message
        token = auth_token_pb2.AuthToken()
        token.ParseFromString(token_bytes)
        
        # Validate required fields
        if not token.HasField('user_id') or not token.HasField('issued_at'):
            print("INVALID")
            return
        
        # Check expiration
        current_time = int(datetime.now().timestamp())
        if token.HasField('expires_at') and token.expires_at < current_time:
            print("INVALID")
            return
        
        # Check admin status
        if token.HasField('is_admin') and token.is_admin == True:
            print("VALID_ADMIN")
            return
        elif token.HasField('role') and token.role == 'admin':
            print("VALID_ADMIN")
            return
        elif token.HasField('privilege_level') and token.privilege_level >= 100:
            print("VALID_ADMIN")
            return
        else:
            print("VALID_USER")
            return
            
    except Exception as e:
        print("INVALID")
        return

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("INVALID")
        sys.exit(1)
    
    validate_token(sys.argv[1])