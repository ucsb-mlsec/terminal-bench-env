#!/usr/bin/env python3

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad
import sys
import json
import os

# Hardcoded AES-128 encryption key (16 bytes)
# In a real application, this would be securely stored and never hardcoded
KEY = b'SecretKey1234567'

def decrypt_token(encrypted_data):
    """
    Decrypts an AES-128-CBC encrypted token.
    
    CBC Mode Decryption Process:
    1. Extract the IV (Initialization Vector) from the first 16 bytes
    2. Use the IV and key to initialize the AES cipher in CBC mode
    3. Decrypt the ciphertext (everything after the IV)
    4. Remove PKCS7 padding from the decrypted plaintext
    
    Args:
        encrypted_data: Raw binary data containing IV + ciphertext
        
    Returns:
        Decrypted plaintext as a string, or None if decryption fails
    """
    try:
        # Extract IV from the first 16 bytes
        iv = encrypted_data[:16]
        
        # Extract the actual ciphertext (everything after the IV)
        ciphertext = encrypted_data[16:]
        
        # Create AES cipher in CBC mode with the extracted IV
        cipher = AES.new(KEY, AES.MODE_CBC, iv)
        
        # Decrypt the ciphertext
        decrypted_padded = cipher.decrypt(ciphertext)
        
        # Remove PKCS7 padding
        decrypted = unpad(decrypted_padded, AES.block_size)
        
        # Return as string
        return decrypted.decode('utf-8')
        
    except ValueError as e:
        print(f"[ERROR] Padding error during decryption: {e}", file=sys.stderr)
        print("[INFO] This might indicate corrupted data or incorrect key", file=sys.stderr)
        # Try to return without unpadding for analysis
        try:
            return decrypted_padded.decode('utf-8', errors='replace')
        except:
            return None
    except Exception as e:
        print(f"[ERROR] Decryption failed: {e}", file=sys.stderr)
        return None

def encrypt_token(plaintext):
    """
    Encrypts plaintext using AES-128-CBC.
    Used to create the initial test token.
    
    Args:
        plaintext: String to encrypt
        
    Returns:
        IV + ciphertext as bytes
    """
    # Generate a random IV
    iv = os.urandom(16)
    
    # Create cipher
    cipher = AES.new(KEY, AES.MODE_CBC, iv)
    
    # Pad the plaintext to block size
    padded = pad(plaintext.encode('utf-8'), AES.block_size)
    
    # Encrypt
    ciphertext = cipher.encrypt(padded)
    
    # Return IV + ciphertext
    return iv + ciphertext

def main():
    if len(sys.argv) < 2:
        print("Usage: python decrypt_service.py <encrypted_token_file>")
        print("\nExample:")
        print("  python decrypt_service.py user_token.enc")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # Read the encrypted token file
        with open(filename, 'rb') as f:
            encrypted_data = f.read()
        
        print(f"[INFO] Read {len(encrypted_data)} bytes from {filename}")
        print(f"[INFO] IV (first 16 bytes): {encrypted_data[:16].hex()}")
        print(f"[INFO] Ciphertext length: {len(encrypted_data[16:])} bytes")
        print()
        
        # Decrypt the token
        decrypted = decrypt_token(encrypted_data)
        
        if decrypted is None:
            print("[FAILED] Could not decrypt token")
            sys.exit(1)
        
        print("[SUCCESS] Decrypted token:")
        print("-" * 60)
        print(decrypted)
        print("-" * 60)
        print()
        
        # Try to parse as JSON
        try:
            data = json.loads(decrypted)
            print("[INFO] Valid JSON structure:")
            print(json.dumps(data, indent=2))
            print()
            
            # Check for admin role
            if 'role' in data:
                role = data['role']
                if role == 'admin':
                    print("=" * 60)
                    print("[SUCCESS] ADMIN ACCESS GRANTED!")
                    print("=" * 60)
                else:
                    print(f"[INFO] Current role: {role}")
            
        except json.JSONDecodeError as e:
            print(f"[WARNING] Not valid JSON: {e}")
            print("[INFO] Decrypted data may be corrupted")
            
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # If creating initial token for testing
    if len(sys.argv) > 2 and sys.argv[1] == "--create":
        plaintext = '{"username":"alice","role":"user","uid":1001}'
        encrypted = encrypt_token(plaintext)
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "user_token.enc"
        with open(output_file, 'wb') as f:
            f.write(encrypted)
        
        print(f"[INFO] Created encrypted token: {output_file}")
        print(f"[INFO] Plaintext: {plaintext}")
        print(f"[INFO] Encrypted length: {len(encrypted)} bytes")
    else:
        main()
```

```python
#!/usr/bin/env python3

"""
CBC Bit Flipping Attack - Privilege Escalation Exploit

This script exploits the lack of integrity checking in CBC mode encryption
to modify encrypted data without knowing the encryption key.

CBC Bit Flipping Vulnerability:
- In CBC mode, flipping a bit in ciphertext block N causes:
  1. Random corruption in plaintext block N
  2. The same bit flip in plaintext block N+1
  
- This allows controlled modification of block N+1 by sacrificing block N
"""

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import os

# The key for creating the original token (in real attack, we don't know this)
KEY = b'SecretKey1234567'

def create_original_token():
    """Create the original user token that we'll modify"""
    plaintext = '{"username":"alice","role":"user","uid":1001}'
    
    # Generate IV
    iv = os.urandom(16)
    
    # Create cipher and encrypt
    cipher = AES.new(KEY, AES.MODE_CBC, iv)
    padded = pad(plaintext.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded)
    
    return iv + ciphertext, plaintext

def perform_bit_flip_attack(encrypted_data, original_plaintext):
    """
    Perform CBC bit flipping attack to change "user" to "admin"
    
    Strategy:
    1. Identify which block contains "user" in plaintext
    2. Modify the previous ciphertext block to flip bits
    3. The target block will have the desired changes
    4. The previous block will be corrupted (but we can work around this)
    """
    
    # Make a mutable copy
    modified = bytearray(encrypted_data)
    
    # Original and target strings
    original_str = '"role":"user"'
    target_str = '"role":"admin"'
    
    # Find position of "user" in plaintext
    # Plaintext: {"username":"alice","role":"user","uid":1001}
    # Position of 'u' in "user" is at index 29
    
    plaintext_pos = original_plaintext.index('user')
    print(f"[INFO] 'user' found at plaintext position: {plaintext_pos}")
    
    # Calculate block positions
    # IV is first 16 bytes, then ciphertext blocks
    # Block 0: positions 0-15 in plaintext
    # Block 1: positions 16-31 in plaintext
    # Block 2: positions 32-47 in plaintext
    
    target_block = plaintext_pos // 16
    position_in_block = plaintext_pos % 16
    
    print(f"[INFO] Target is in plaintext block {target_block}, position {position_in_block}")
    
    # To modify block N, we flip bits in ciphertext block N-1
    # Ciphertext structure: [IV][Block0][Block1][Block2]...
    # IV is at bytes 0-15
    # Block 0 is at bytes 16-31
    # Block 1 is at bytes 32-47
    
    # The ciphertext block we need to modify is at:
    ciphertext_block_to_modify = target_block  # 0-indexed in ciphertext
    ciphertext_byte_offset = 16 + (ciphertext_block_to_modify * 16) + position_in_block
    
    print(f"[INFO] Modifying ciphertext starting at byte {ciphertext_byte_offset}")
    
    # XOR operation: 
    # original_plaintext_byte XOR ciphertext_byte XOR target_byte = new_plaintext_byte
    # So: new_ciphertext_byte = ciphertext_byte XOR original_byte XOR target_byte
    
    for i, (orig_char, target_char) in enumerate(zip('user', 'admin')):
        offset = ciphertext_byte_offset + i
        
        # XOR the difference
        xor_value = ord(orig_char) ^ ord(target_char)
        modified[offset] ^= xor_value
        
        print(f"[INFO] Flipping bit at offset {offset}: '{orig_char}' -> '{target_char}' (XOR with 0x{xor_value:02x})")
    
    # Handle the length difference: "user" (4 chars) vs "admin" (5 chars)
    # We need to also modify one more character after "user"
    # The character after "user" is '"', we'll keep it as is by modifying padding/structure
    
    # Actually, let's reconsider: we need exact same length
    # Let's check the JSON structure more carefully
    
    print(f"\n[INFO] Original plaintext: {original_plaintext}")
    print(f"[INFO] Original length: {len(original_plaintext)}")
    print(f"[INFO] Target would be:    " + original_plaintext.replace('user', 'admin'))
    
    return bytes(modified)

def main():
    print("=" * 70)
    print("CBC Bit Flipping Attack - Privilege Escalation")
    print("=" * 70)
    print()
    
    # Step 1: Create the original encrypted token
    print("[STEP 1] Creating original user token...")
    encrypted_data, plaintext = create_original_token()
    
    # Save original token
    os.makedirs('/tmp/solution', exist_ok=True)
    with open('/tmp/solution/user_token.enc', 'wb') as f:
        f.write(encrypted_data)
    print(f"[INFO] Original plaintext: {plaintext}")
    print(f"[INFO] Encrypted length: {len(encrypted_data)} bytes")
    print()
    
    # Step 2: Perform bit flipping attack
    print("[STEP 2] Performing bit flipping attack...")
    modified_encrypted = perform_bit_flip_attack(encrypted_data, plaintext)
    print()
    
    # Step 3: Save the modified token
    print("[STEP 3] Saving modified admin token...")
    with open('/tmp/solution/admin_token.enc', 'wb') as f:
        f.write(modified_encrypted)
    print("[SUCCESS] Modified token saved to /tmp/solution/admin_token.enc")
    print()
    
    # Step 4: Verify (decrypt and check)
    print("[STEP 4] Verifying the attack...")
    print("Run: python decrypt_service.py /tmp/solution/admin_token.enc")
    print()
    
    print("=" * 70)
    print("Attack complete! The modified token should grant admin access.")
    print("Note: The block before 'admin' will be corrupted, but the JSON")
    print("parser should still be able to extract the role field.")
    print("=" * 70)

if __name__ == "__main__":
    main()