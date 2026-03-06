#!/usr/bin/env python3

import sys
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# Hardcoded AES-128 key (16 bytes)
KEY = b'SecretKey1234567'

def read_binary_file(filepath):
    """Read binary data from file."""
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def decrypt_cookie(ciphertext, iv, key):
    """Decrypt the cookie using AES-128 CBC mode."""
    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(ciphertext)
        # Remove PKCS7 padding
        plaintext = unpad(decrypted, AES.block_size)
        return plaintext
    except ValueError as e:
        print(f"Error during decryption (padding issue): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error during decryption: {e}", file=sys.stderr)
        return None

def check_admin_access(plaintext):
    """Check if the plaintext indicates admin privileges."""
    if plaintext is None:
        return False
    
    plaintext_str = plaintext.decode('utf-8', errors='ignore')
    
    # Check for various admin privilege indicators
    admin_indicators = [
        'role=admin',
        'admin=true',
        'privilege=admin',
        'access=admin',
        'level=admin',
        'type=admin',
        'isadmin=true',
        'is_admin=true'
    ]
    
    plaintext_lower = plaintext_str.lower()
    for indicator in admin_indicators:
        if indicator in plaintext_lower:
            return True
    
    return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 decrypt_and_check.py <cookie_file> <iv_file>", file=sys.stderr)
        sys.exit(1)
    
    cookie_file = sys.argv[1]
    iv_file = sys.argv[2]
    
    # Read the encrypted cookie and IV
    ciphertext = read_binary_file(cookie_file)
    iv = read_binary_file(iv_file)
    
    # Validate IV length
    if len(iv) != 16:
        print(f"Error: IV must be 16 bytes, got {len(iv)} bytes", file=sys.stderr)
        sys.exit(1)
    
    # Validate ciphertext is multiple of block size
    if len(ciphertext) % 16 != 0:
        print(f"Error: Ciphertext length must be multiple of 16 bytes, got {len(ciphertext)} bytes", file=sys.stderr)
        sys.exit(1)
    
    # Decrypt the cookie
    plaintext = decrypt_cookie(ciphertext, iv, KEY)
    
    if plaintext is None:
        print("Failed to decrypt cookie", file=sys.stderr)
        sys.exit(1)
    
    # Print decrypted plaintext for debugging
    try:
        plaintext_str = plaintext.decode('utf-8', errors='replace')
        print("Decrypted plaintext:")
        print(plaintext_str)
    except Exception as e:
        print(f"Decrypted plaintext (hex): {plaintext.hex()}")
    
    # Check for admin access
    if check_admin_access(plaintext):
        print("\n[SUCCESS] Admin access granted!")
        sys.exit(0)
    else:
        print("\n[FAILED] No admin access")
        sys.exit(1)

if __name__ == '__main__':
    main()