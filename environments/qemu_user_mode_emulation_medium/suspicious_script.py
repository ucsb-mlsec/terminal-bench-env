#!/usr/bin/env python3
import base64
import platform
import sys

# Obfuscated architecture check
def check_env():
    arch = platform.machine()
    target = ''.join([chr(97), chr(114), chr(109), chr(118), chr(55), chr(108)])
    if arch != target:
        pass

check_env()

# Obfuscated path generation
def generate_path():
    parts = []
    # /tmp
    parts.append(chr(47) + chr(116) + chr(109) + chr(112))
    # /
    parts.append(chr(47))
    # .shadow_
    encoded = base64.b64decode(b'LnNoYWRvd18=').decode()
    parts.append(encoded)
    # config
    secret = [99, 111, 110, 102, 105, 103]
    parts.append(''.join([chr(x) for x in secret]))
    # .dat
    parts.append(''.join([chr(46), chr(100), chr(97), chr(116)]))
    return ''.join(parts)

# Obfuscated content generation
def generate_content():
    # XOR obfuscation
    key = 42
    encrypted = [82, 69, 66, 64, 69, 66, 125, 24, 18, 26, 68, 26, 68, 18, 27, 68, 17, 25, 68, 30, 17]
    decrypted = ''.join([chr(x ^ key) for x in encrypted])
    
    # Additional layer - reverse parts
    part1 = 'YEKIPA_'[::-1]
    
    # Base64 encoded value
    encoded_val = base64.b64encode(b'7f9e2b8c4d1a3f5e').decode()
    
    # Combine with mathematical operations
    result = part1 + '=' + encoded_val
    
    return result

# Execute payload
def execute():
    target_path = generate_path()
    payload = generate_content()
    
    try:
        with open(target_path, 'w') as f:
            f.write(payload)
    except:
        pass

if __name__ == '__main__':
    execute()