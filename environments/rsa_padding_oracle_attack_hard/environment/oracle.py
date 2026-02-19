#!/usr/bin/env python3

import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from Crypto.Random import get_random_bytes

# Generate or load RSA key pair
KEY_PATH = '/workspace/private_key.pem'
PUB_KEY_PATH = '/workspace/public_key.pem'

if os.path.exists(KEY_PATH):
    with open(KEY_PATH, 'rb') as f:
        private_key = RSA.import_key(f.read())
else:
    # Generate new 2048-bit RSA key pair
    private_key = RSA.generate(2048)
    
    # Save private key
    with open(KEY_PATH, 'wb') as f:
        f.write(private_key.export_key('PEM'))
    
    # Save public key
    public_key = private_key.publickey()
    with open(PUB_KEY_PATH, 'wb') as f:
        f.write(public_key.export_key('PEM'))

def check_padding(ciphertext_bytes: bytes) -> bool:
    """
    Oracle function that checks if the PKCS#1 v1.5 padding is valid.
    
    Args:
        ciphertext_bytes: The ciphertext to check (as bytes)
    
    Returns:
        True if padding is valid, False otherwise
    """
    try:
        # Get the key size in bytes
        key_size = private_key.size_in_bytes()
        
        # Ensure ciphertext is the correct length
        if len(ciphertext_bytes) != key_size:
            return False
        
        # Convert ciphertext bytes to integer
        c = int.from_bytes(ciphertext_bytes, byteorder='big')
        
        # Perform raw RSA decryption: m = c^d mod n
        m = pow(c, private_key.d, private_key.n)
        
        # Convert back to bytes with proper padding
        m_bytes = m.to_bytes(key_size, byteorder='big')
        
        # Check PKCS#1 v1.5 padding format
        # Format: 0x00 || 0x02 || PS || 0x00 || M
        # where PS is at least 8 random non-zero bytes
        
        # Check first byte is 0x00
        if m_bytes[0] != 0x00:
            return False
        
        # Check second byte is 0x02 (for encryption)
        if m_bytes[1] != 0x02:
            return False
        
        # Find the 0x00 separator after padding string
        separator_found = False
        separator_index = -1
        
        # Start from byte 2 (after 0x00 0x02)
        for i in range(2, len(m_bytes)):
            if m_bytes[i] == 0x00:
                separator_found = True
                separator_index = i
                break
        
        # Must find separator and padding must be at least 8 bytes
        if not separator_found or separator_index < 10:  # 2 (header) + 8 (min padding)
            return False
        
        # Check that all padding bytes (between position 2 and separator) are non-zero
        for i in range(2, separator_index):
            if m_bytes[i] == 0x00:
                return False
        
        return True
        
    except Exception as e:
        return False

# Create encrypted token file if it doesn't exist
ENCRYPTED_TOKEN_PATH = '/workspace/encrypted_token.bin'
if not os.path.exists(ENCRYPTED_TOKEN_PATH):
    # Create a sample authentication token
    plaintext = b"AUTH_TOKEN_987654321_SECURE"
    
    # Encrypt with PKCS#1 v1.5
    cipher = PKCS1_v1_5.new(private_key.publickey())
    
    # For proper encryption, we need to manually pad and encrypt
    # or use the cipher encrypt method
    key_size = private_key.size_in_bytes()
    
    # Manually create PKCS#1 v1.5 padded message
    ps_len = key_size - len(plaintext) - 3  # 3 bytes for 0x00 0x02 and separator 0x00
    if ps_len < 8:
        raise ValueError("Message too long")
    
    # Generate random non-zero padding
    ps = bytes([b for b in get_random_bytes(ps_len) if b != 0][:ps_len])
    while len(ps) < ps_len or 0 in ps:
        ps = bytes([b if b != 0 else 1 for b in get_random_bytes(ps_len)])[:ps_len]
    
    # Construct padded message: 0x00 || 0x02 || PS || 0x00 || M
    padded_message = b'\x00\x02' + ps + b'\x00' + plaintext
    
    # Convert to integer and encrypt
    m = int.from_bytes(padded_message, byteorder='big')
    c = pow(m, private_key.e, private_key.n)
    ciphertext = c.to_bytes(key_size, byteorder='big')
    
    # Save encrypted token
    with open(ENCRYPTED_TOKEN_PATH, 'wb') as f:
        f.write(ciphertext)