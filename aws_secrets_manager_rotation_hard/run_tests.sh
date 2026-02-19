def test_secret_step(secret_arn, token):
    """Test that new secret credentials work - FIXED VERSION"""
    import json
    import sys
    sys.path.insert(0, '/opt/rotation')
    import secrets_mock
    import db_mock
    
    # Get the secret value for the specified version
    secret = secrets_mock.get_secret_value(secret_arn, version_id=token)
    
    # Problem 3 Fix: Verify the secret has the AWSPENDING staging label
    if 'VersionStages' not in secret or 'AWSPENDING' not in secret['VersionStages']:
        print(f"Secret does not have AWSPENDING label")
        return False
    
    # Parse the secret data
    secret_data = json.loads(secret['SecretString'])
    
    # Problem 1 Fix: Use correct field name 'password' instead of 'pass'
    conn_string = f"postgresql://{secret_data['username']}:{secret_data['password']}@{secret_data['host']}:{secret_data['port']}/mydb"
    
    # Problem 2 Fix: Return False when connection fails
    try:
        db_mock.test_connection(conn_string)
    except ConnectionError as e:
        print(f"Connection failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during connection test: {e}")
        return False
    
    # Return True only when connection succeeds and AWSPENDING label exists
    return True