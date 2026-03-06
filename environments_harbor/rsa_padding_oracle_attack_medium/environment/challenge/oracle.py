#!/usr/bin/env python3

import sys
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.backends import default_backend

def check_pkcs1_v15_padding(decrypted_data):
    """
    Check if the decrypted data has valid PKCS#1 v1.5 padding.
    Format: 0x00 || 0x02 || PS || 0x00 || M
    where PS is at least 8 non-zero bytes
    """
    try:
        if len(decrypted_data) < 11:  # Minimum: 0x00 + 0x02 + 8 bytes PS + 0x00
            return False
        
        # Check first byte is 0x00
        if decrypted_data[0] != 0x00:
            return False
        
        # Check second byte is 0x02
        if decrypted_data[1] != 0x02:
            return False
        
        # Find the 0x00 separator after the padding
        separator_index = None
        for i in range(2, len(decrypted_data)):
            if decrypted_data[i] == 0x00:
                separator_index = i
                break
        
        # Check if separator was found
        if separator_index is None:
            return False
        
        # Check that there are at least 8 padding bytes (between index 2 and separator)
        padding_length = separator_index - 2
        if padding_length < 8:
            return False
        
        # Check that all padding bytes are non-zero
        for i in range(2, separator_index):
            if decrypted_data[i] == 0x00:
                return False
        
        return True
    except:
        return False

def main():
    try:
        if len(sys.argv) != 2:
            with open('/tmp/oracle_response.txt', 'w') as f:
                f.write('INVALID')
            return
        
        ciphertext_file = sys.argv[1]
        
        # Read the ciphertext
        with open(ciphertext_file, 'rb') as f:
            ciphertext = f.read()
        
        # Load the private key
        with open('/challenge/private_key.pem', 'rb') as f:
            private_key = load_pem_private_key(f.read(), password=None, backend=default_backend())
        
        # Get the key parameters for raw RSA operation
        private_numbers = private_key.private_numbers()
        n = private_numbers.public_numbers.n
        d = private_numbers.d
        
        # Convert ciphertext to integer
        c = int.from_bytes(ciphertext, byteorder='big')
        
        # Perform raw RSA decryption: m = c^d mod n
        m = pow(c, d, n)
        
        # Convert back to bytes with proper padding
        key_size_bytes = (private_key.key_size + 7) // 8
        decrypted_data = m.to_bytes(key_size_bytes, byteorder='big')
        
        # Check PKCS#1 v1.5 padding
        if check_pkcs1_v15_padding(decrypted_data):
            with open('/tmp/oracle_response.txt', 'w') as f:
                f.write('VALID')
        else:
            with open('/tmp/oracle_response.txt', 'w') as f:
                f.write('INVALID')
    
    except Exception as e:
        # Any error is treated as invalid padding
        with open('/tmp/oracle_response.txt', 'w') as f:
            f.write('INVALID')

if __name__ == '__main__':
    main()