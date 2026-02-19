import re


class ConnectionError(Exception):
    """Exception raised when database connection fails"""
    pass


def parse_connection_string(connection_string):
    """Parse PostgreSQL connection string and extract components"""
    pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
    match = re.match(pattern, connection_string)
    
    if not match:
        raise ConnectionError("Malformed connection string")
    
    username, password, host, port, database = match.groups()
    return {
        'username': username,
        'password': password,
        'host': host,
        'port': port,
        'database': database
    }


def test_connection(connection_string):
    """
    Simulate PostgreSQL connection testing
    
    Args:
        connection_string: PostgreSQL connection string
        
    Returns:
        True if connection succeeds
        
    Raises:
        ConnectionError: If connection fails
    """
    try:
        creds = parse_connection_string(connection_string)
    except ConnectionError:
        raise ConnectionError("Invalid connection string format")
    
    password = creds['password']
    username = creds['username']
    
    # Simulate valid passwords
    valid_passwords = ['valid_password', 'correct_pass']
    
    # Simulate invalid passwords
    invalid_passwords = ['invalid_password', 'wrong_pass', 'bad_pass', 'expired_pass']
    
    if password in valid_passwords:
        print(f"Connection successful for user {username}")
        return True
    elif password in invalid_passwords:
        raise ConnectionError(f"Authentication failed for user {username}")
    else:
        # Unknown password - treat as invalid
        raise ConnectionError(f"Authentication failed for user {username}")