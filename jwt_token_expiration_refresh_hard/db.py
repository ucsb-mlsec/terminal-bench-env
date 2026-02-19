import sqlite3
from datetime import datetime

DATABASE = '/app/auth.db'

def get_db_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def validate_user(username, password):
    """
    Validates user credentials.
    Returns user_id if valid, None otherwise.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id FROM users WHERE username = ? AND password = ?',
            (username, password)
        )
        result = cursor.fetchone()
        if result:
            return result['id']
        return None
    finally:
        conn.close()

def store_refresh_token(user_id, token_jti, parent_jti=None):
    """
    Stores a refresh token in the database.
    VULNERABILITY: This function exists but is NOT called by auth_api or tokens modules.
    
    Args:
        user_id: The ID of the user this token belongs to
        token_jti: The JTI (unique identifier) of the refresh token
        parent_jti: The JTI of the parent token (for token family tracking)
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO refresh_tokens (jti, user_id, parent_jti, created_at, used)
               VALUES (?, ?, ?, ?, 0)''',
            (token_jti, user_id, parent_jti, datetime.utcnow())
        )
        conn.commit()
    finally:
        conn.close()

def invalidate_token(token_jti):
    """
    Marks a token as used in the database.
    VULNERABILITY: This function exists but is NOT called when tokens are refreshed.
    
    Args:
        token_jti: The JTI of the token to invalidate
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE refresh_tokens SET used = 1 WHERE jti = ?',
            (token_jti,)
        )
        conn.commit()
    finally:
        conn.close()

def check_token_valid(token_jti):
    """
    Checks if a token is valid (exists and not marked as used).
    VULNERABILITY: This function exists but is NOT called during token validation.
    
    Args:
        token_jti: The JTI of the token to check
        
    Returns:
        True if token is valid (exists and not used), False otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT used FROM refresh_tokens WHERE jti = ?',
            (token_jti,)
        )
        result = cursor.fetchone()
        
        if result is None:
            # Token doesn't exist in database
            return False
        
        # Token is valid only if it exists and used = 0
        return result['used'] == 0
    finally:
        conn.close()

def get_token_info(token_jti):
    """
    Retrieves information about a token.
    
    Args:
        token_jti: The JTI of the token
        
    Returns:
        Dictionary with token information or None if not found
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT jti, user_id, parent_jti, created_at, used FROM refresh_tokens WHERE jti = ?',
            (token_jti,)
        )
        result = cursor.fetchone()
        if result:
            return dict(result)
        return None
    finally:
        conn.close()

def invalidate_token_family(token_jti):
    """
    Invalidates an entire token family (for breach detection).
    
    Args:
        token_jti: The JTI of any token in the family
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # First, find the user_id for this token
        cursor.execute('SELECT user_id FROM refresh_tokens WHERE jti = ?', (token_jti,))
        result = cursor.fetchone()
        
        if result:
            user_id = result['user_id']
            # Invalidate all tokens for this user
            cursor.execute(
                'UPDATE refresh_tokens SET used = 1 WHERE user_id = ?',
                (user_id,)
            )
            conn.commit()
    finally:
        conn.close()