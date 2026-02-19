#!/usr/bin/env python3

"""
LDAP Authentication Module for Web Application
This module handles user authentication against an LDAP directory server.
WARNING: This code contains security vulnerabilities for educational purposes.
"""

from ldap3 import Server, Connection, ALL, SUBTREE
from ldap3.core.exceptions import LDAPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LDAP Server Configuration
LDAP_HOST = 'ldap://localhost:389'
BASE_DN = 'dc=company,dc=com'
LDAP_USER_DN = 'cn=admin,dc=company,dc=com'
LDAP_PASSWORD = 'admin_password'


def get_ldap_connection():
    """
    Establishes connection to LDAP server
    Returns Connection object or None on failure
    """
    try:
        server = Server(LDAP_HOST, get_info=ALL)
        conn = Connection(server, user=LDAP_USER_DN, password=LDAP_PASSWORD, auto_bind=True)
        return conn
    except LDAPException as e:
        logger.error(f"Failed to connect to LDAP server: {e}")
        return None


def authenticate_user(username, password):
    """
    Authenticates user against LDAP directory
    
    Args:
        username: User's login username
        password: User's password
        
    Returns:
        bool: True if authentication successful, False otherwise
        
    Authentication Flow:
    1. Connect to LDAP server
    2. Construct search filter with username and password
    3. Search for matching entry in directory
    4. Return True if entry found, False otherwise
    """
    
    # Establish LDAP connection
    conn = get_ldap_connection()
    if not conn:
        logger.error("Could not establish LDAP connection")
        return False
    
    try:
        # VULNERABILITY: Direct string concatenation without sanitization
        # This allows LDAP injection attacks through the username parameter
        # The username input is directly embedded into the LDAP filter
        ldap_filter = f'(&(objectClass=person)(uid={username})(userPassword={password}))'
        
        logger.info(f"Executing LDAP search with filter: {ldap_filter}")
        
        # Search LDAP directory for matching user
        search_result = conn.search(
            search_base=BASE_DN,
            search_filter=ldap_filter,
            search_scope=SUBTREE,
            attributes=['uid', 'cn', 'mail']
        )
        
        if search_result and len(conn.entries) > 0:
            # User found - authentication successful
            logger.info(f"Authentication successful for user: {username}")
            logger.debug(f"Found {len(conn.entries)} matching entries")
            conn.unbind()
            return True
        else:
            # No matching user found
            logger.warning(f"Authentication failed for user: {username}")
            conn.unbind()
            return False
            
    except LDAPException as e:
        # Handle LDAP-specific errors
        logger.error(f"LDAP error during authentication: {e}")
        if conn:
            conn.unbind()
        return False
    except Exception as e:
        # Handle general errors
        logger.error(f"Unexpected error during authentication: {e}")
        if conn:
            conn.unbind()
        return False


def validate_credentials(username, password):
    """
    Wrapper function for credential validation
    Adds basic input validation (but still vulnerable to injection)
    """
    # Basic validation - check for empty inputs
    if not username or not password:
        logger.warning("Empty username or password provided")
        return False
    
    # Call main authentication function
    return authenticate_user(username, password)


# Example usage
if __name__ == "__main__":
    # Test authentication
    test_user = "jdoe"
    test_pass = "password123"
    
    result = authenticate_user(test_user, test_pass)
    print(f"Authentication result for {test_user}: {result}")