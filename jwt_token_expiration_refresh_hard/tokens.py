import jwt
import datetime
import secrets
import uuid
from typing import Optional, Dict, Any

SECRET_KEY = 'super-secret-key-change-in-production'

def generate_access_token(user_id: int) -> str:
    """
    Generate a short-lived access token for API requests.
    
    Args:
        user_id: The user's unique identifier
        
    Returns:
        Encoded JWT access token string
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=15),
        'iat': datetime.datetime.utcnow(),
        'type': 'access'
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token


def generate_refresh_token(user_id: int) -> str:
    """
    Generate a long-lived refresh token for obtaining new access tokens.
    
    Args:
        user_id: The user's unique identifier
        
    Returns:
        Encoded JWT refresh token string
        
    VULNERABILITY: Does not properly track or store token information for rotation.
    The jti (JWT ID) is generated but not stored in the database, making it impossible
    to track which tokens have been used or implement proper token rotation.
    """
    jti = str(uuid.uuid4())
    
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7),
        'iat': datetime.datetime.utcnow(),
        'jti': jti,
        'type': 'refresh'
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token


def validate_access_token(token: str) -> Optional[int]:
    """
    Validate an access token and extract the user_id.
    
    Args:
        token: The JWT access token to validate
        
    Returns:
        user_id if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        
        if payload.get('type') != 'access':
            return None
            
        return payload.get('user_id')
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def validate_refresh_token(token: str) -> Optional[int]:
    """
    Validate a refresh token and extract the user_id.
    
    Args:
        token: The JWT refresh token to validate
        
    Returns:
        user_id if token is valid, None otherwise
        
    VULNERABILITY: Does not check if token has been used before or invalidated.
    This allows an attacker who steals a refresh token to reuse it indefinitely,
    even after the legitimate user has used it to get a new token. No database
    checks are performed to verify token status.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        
        if payload.get('type') != 'refresh':
            return None
            
        return payload.get('user_id')
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode a token without validation (for extracting jti and other claims).
    
    Args:
        token: The JWT token to decode
        
    Returns:
        Decoded payload dict if successful, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.InvalidTokenError:
        return None