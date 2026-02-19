#!/usr/bin/env python3

import hashlib
import secrets
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, urlencode
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging for audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/agent/access.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OAuthServer')


class OAuthServerConfig:
    """Configuration for OAuth 2.0 Authorization Server"""
    
    def __init__(self):
        self.token_expiry = 3600  # 1 hour
        self.authorization_code_expiry = 600  # 10 minutes
        self.supported_grant_types = ['authorization_code', 'implicit']
        self.supported_response_types = ['code', 'token']
        
        # Registered OAuth clients with their configurations
        self.clients = {
            'client_app_001': {
                'client_id': 'client_app_001',
                'client_secret': hashlib.sha256(b'secret_001').hexdigest(),
                'name': 'TrustedApp',
                'redirect_uris': [
                    'https://trusted-app.com/callback',
                    'https://trusted-app.com/oauth/callback'
                ],
                'allowed_scopes': ['read', 'write', 'profile']
            },
            'client_app_002': {
                'client_id': 'client_app_002',
                'client_secret': hashlib.sha256(b'secret_002').hexdigest(),
                'name': 'MobileApp',
                'redirect_uris': [
                    'https://mobile-app.example.com/auth/callback'
                ],
                'allowed_scopes': ['read', 'profile']
            },
            'client_app_003': {
                'client_id': 'client_app_003',
                'client_secret': hashlib.sha256(b'secret_003').hexdigest(),
                'name': 'PartnerService',
                'redirect_uris': [
                    'https://partner.service.io/oauth',
                    'https://partner.service.io/oauth/return'
                ],
                'allowed_scopes': ['read', 'write']
            },
            'client_app_004': {
                'client_id': 'client_app_004',
                'client_secret': hashlib.sha256(b'secret_004').hexdigest(),
                'name': 'EnterpriseApp',
                'redirect_uris': [
                    'https://enterprise.corp.net/callback'
                ],
                'allowed_scopes': ['read', 'write', 'admin']
            }
        }
        
        # Active user sessions
        self.user_sessions = {}
        
        # Authorization codes storage
        self.auth_codes = {}
        
        # Access tokens storage
        self.access_tokens = {}


class OAuthServer:
    """OAuth 2.0 Authorization Server with redirect URI validation"""
    
    def __init__(self, config: OAuthServerConfig):
        self.config = config
        self.logger = logger
        
    def get_client_config(self, client_id: str) -> Optional[Dict]:
        """Retrieve client configuration by client ID"""
        client = self.config.clients.get(client_id)
        if client:
            self.logger.info(f"Client configuration retrieved for: {client_id}")
        else:
            self.logger.warning(f"Client not found: {client_id}")
        return client
    
    def validate_redirect_uri(self, client_id: str, redirect_uri: str) -> bool:
        """
        Validate redirect URI against registered URIs for the client.
        
        Security Note: This is a critical security function. We need to ensure
        that redirect URIs match registered URIs to prevent open redirect attacks.
        Implementation follows OAuth 2.0 security best practices.
        """
        client = self.get_client_config(client_id)
        if not client:
            self.logger.error(f"Validation failed: Unknown client {client_id}")
            return False
        
        registered_uris = client.get('redirect_uris', [])
        if not registered_uris:
            self.logger.error(f"No registered redirect URIs for client {client_id}")
            return False
        
        self.logger.info(f"Validating redirect_uri: {redirect_uri} for client: {client_id}")
        
        # VULNERABILITY 1: Using 'in' operator in wrong order - allows substring bypass
        # Attacker can use: https://evil.com?url=https://trusted-app.com/callback
        for registered_uri in registered_uris:
            if registered_uri in redirect_uri:
                self.logger.info(f"Redirect URI validated (substring match): {redirect_uri}")
                return True
        
        # VULNERABILITY 2: Using startswith without proper validation
        # This allows path traversal and additional paths
        # e.g., https://trusted-app.com/callback.evil.com will pass
        for registered_uri in registered_uris:
            # Strip query parameters for comparison - but this is done incorrectly
            base_registered = registered_uri.split('?')[0]
            if redirect_uri.startswith(base_registered):
                self.logger.info(f"Redirect URI validated (prefix match): {redirect_uri}")
                return True
        
        # VULNERABILITY 3: Case insensitive comparison allows bypass
        # Attacker can use mixed case to bypass validation
        for registered_uri in registered_uris:
            if redirect_uri.lower().startswith(registered_uri.lower()):
                self.logger.info(f"Redirect URI validated (case-insensitive): {redirect_uri}")
                return True
        
        # VULNERABILITY 4: Regex pattern matching that's too permissive
        # Trying to validate domain but allowing subdomains incorrectly
        for registered_uri in registered_uris:
            parsed_registered = urlparse(registered_uri)
            parsed_redirect = urlparse(redirect_uri)
            
            # This regex is too permissive - doesn't anchor properly
            # Allows attacker.trusted-app.com to match trusted-app.com
            domain_pattern = re.escape(parsed_registered.netloc)
            # VULNERABILITY: Not anchoring the pattern properly
            if re.search(domain_pattern, parsed_redirect.netloc):
                self.logger.info(f"Redirect URI validated (domain pattern): {redirect_uri}")
                return True
        
        # VULNERABILITY 5: Protocol validation missing
        # Not checking if http vs https, allowing downgrade attacks
        for registered_uri in registered_uris:
            parsed_registered = urlparse(registered_uri)
            parsed_redirect = urlparse(redirect_uri)
            
            # Only comparing netloc and path, ignoring scheme
            if (parsed_registered.netloc == parsed_redirect.netloc and 
                parsed_redirect.path.startswith(parsed_registered.path)):
                self.logger.info(f"Redirect URI validated (path match, protocol ignored): {redirect_uri}")
                return True
        
        # VULNERABILITY 6: Query parameter manipulation not properly validated
        # Allows adding arbitrary query parameters that could be used for attacks
        for registered_uri in registered_uris:
            redirect_base = redirect_uri.split('?')[0]
            registered_base = registered_uri.split('?')[0]
            
            if redirect_base == registered_base:
                # Accepting any query parameters without validation
                self.logger.info(f"Redirect URI validated (base match, params ignored): {redirect_uri}")
                return True
        
        # VULNERABILITY 7: Subdomain wildcard validation done incorrectly
        # Trying to support *.domain.com but implementation is flawed
        for registered_uri in registered_uris:
            parsed_registered = urlparse(registered_uri)
            parsed_redirect = urlparse(redirect_uri)
            
            # Extract base domain incorrectly - doesn't handle all cases
            registered_parts = parsed_registered.netloc.split('.')
            redirect_parts = parsed_redirect.netloc.split('.')
            
            # VULNERABILITY: This allows evil.trusted-app.com to match
            if len(registered_parts) >= 2 and len(redirect_parts) >= 2:
                # Only checking last two parts (domain + TLD)
                if (registered_parts[-2:] == redirect_parts[-2:]):
                    self.logger.info(f"Redirect URI validated (domain suffix match): {redirect_uri}")
                    return True
        
        self.logger.warning(f"Redirect URI validation failed for: {redirect_uri}")
        return False
    
    def create_authorization_code(self, client_id: str, user_id: str, 
                                 redirect_uri: str, scope: str) -> str:
        """Generate authorization code for OAuth flow"""
        code = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(seconds=self.config.authorization_code_expiry)
        
        self.config.auth_codes[code] = {
            'client_id': client_id,
            'user_id': user_id,
            'redirect_uri': redirect_uri,
            'scope': scope,
            'expiry': expiry,
            'used': False
        }
        
        self.logger.info(f"Authorization code created for user: {user_id}, client: {client_id}")
        return code
    
    def create_access_token(self, client_id: str, user_id: str, scope: str) -> Dict:
        """Generate access token"""
        token = secrets.token_urlsafe(48)
        expiry = datetime.now() + timedelta(seconds=self.config.token_expiry)
        
        token_data = {
            'access_token': token,
            'token_type': 'Bearer',
            'expires_in': self.config.token_expiry,
            'scope': scope,
            'client_id': client_id,
            'user_id': user_id,
            'expiry': expiry
        }
        
        self.config.access_tokens[token] = token_data
        self.logger.info(f"Access token created for user: {user_id}, client: {client_id}")
        
        return {
            'access_token': token,
            'token_type': 'Bearer',
            'expires_in': self.config.token_expiry,
            'scope': scope
        }
    
    def authorize(self, client_id: str, redirect_uri: str, user_id: str, 
                 scope: str, response_type: str = 'code', state: str = None) -> Tuple[bool, str]:
        """
        OAuth 2.0 Authorization Endpoint
        
        Handles authorization requests and generates authorization codes or tokens.
        Returns: (success: bool, redirect_url: str)
        """
        self.logger.info(f"Authorization request - client: {client_id}, user: {user_id}, "
                        f"redirect: {redirect_uri}, scope: {scope}")
        
        # Validate client exists
        client = self.get_client_config(client_id)
        if not client:
            error_msg = "invalid_client"
            self.logger.error(f"Authorization failed: {error_msg}")
            return False, f"{redirect_uri}?error={error_msg}"
        
        # Validate response type
        if response_type not in self.config.supported_response_types:
            error_msg = "unsupported_response_type"
            self.logger.error(f"Authorization failed: {error_msg}")
            return False, f"{redirect_uri}?error={error_msg}"
        
        # Validate redirect URI - THIS IS THE CRITICAL SECURITY CHECK
        if not self.validate_redirect_uri(client_id, redirect_uri):
            # Even if validation fails, we still log the attempt
            self.logger.error(f"Redirect URI validation failed for client {client_id}: {redirect_uri}")
            return False, "error: invalid_redirect_uri"
        
        # Validate scope
        requested_scopes = scope.split() if scope else []
        allowed_scopes = client.get('allowed_scopes', [])
        
        for requested_scope in requested_scopes:
            if requested_scope not in allowed_scopes:
                error_msg = "invalid_scope"
                self.logger.warning(f"Authorization warning: {error_msg} - {requested_scope}")
                return False, f"{redirect_uri}?error={error_msg}"
        
        # Generate authorization code or token based on response_type
        if response_type == 'code':
            # Authorization code flow
            auth_code = self.create_authorization_code(client_id, user_id, redirect_uri, scope)
            
            # Build redirect URL with authorization code
            separator = '&' if '?' in redirect_uri else '?'
            redirect_url = f"{redirect_uri}{separator}code={auth_code}"
            if state:
                redirect_url += f"&state={state}"
            
            self.logger.info(f"Authorization successful - code issued to user: {user_id}")
            return True, redirect_url
        
        elif response_type == 'token':
            # Implicit flow - return token directly
            token_data = self.create_access_token(client_id, user_id, scope)
            
            # Build redirect URL with access token in fragment
            separator = '&' if '#' in redirect_uri else '#'
            redirect_url = f"{redirect_uri}{separator}access_token={token_data['access_token']}"
            redirect_url += f"&token_type={token_data['token_type']}"
            redirect_url += f"&expires_in={token_data['expires_in']}"
            if scope:
                redirect_url += f"&scope={scope}"
            if state:
                redirect_url += f"&state={state}"
            
            self.logger.info(f"Authorization successful - token issued to user: {user_id}")
            return True, redirect_url
        
        return False, f"{redirect_uri}?error=server_error"
    
    def exchange_code_for_token(self, client_id: str, client_secret: str, 
                               code: str, redirect_uri: str) -> Optional[Dict]:
        """
        Token endpoint - exchange authorization code for access token
        """
        self.logger.info(f"Token exchange request - client: {client_id}, code: {code}")
        
        # Validate client credentials
        client = self.get_client_config(client_id)
        if not client or client['client_secret'] != client_secret:
            self.logger.error(f"Token exchange failed: invalid client credentials")
            return None
        
        # Validate authorization code
        if code not in self.config.auth_codes:
            self.logger.error(f"Token exchange failed: invalid authorization code")
            return None
        
        code_data = self.config.auth_codes[code]
        
        # Check if code is expired
        if datetime.now() > code_data['expiry']:
            self.logger.error(f"Token exchange failed: authorization code expired")
            return None
        
        # Check if code was already used
        if code_data['used']:
            self.logger.error(f"Token exchange failed: authorization code already used")
            return None
        
        # Validate client_id matches
        if code_data['client_id'] != client_id:
            self.logger.error(f"Token exchange failed: client_id mismatch")
            return None
        
        # Validate redirect_uri matches
        if code_data['redirect_uri'] != redirect_uri:
            self.logger.error(f"Token exchange failed: redirect_uri mismatch")
            return None
        
        # Mark code as used
        code_data['used'] = True
        
        # Create access token
        token_data = self.create_access_token(
            client_id, 
            code_data['user_id'], 
            code_data['scope']
        )
        
        self.logger.info(f"Token exchange successful for user: {code_data['user_id']}")
        return token_data
    
    def validate_access_token(self, token: str) -> Optional[Dict]:
        """Validate an access token and return token data"""
        if token not in self.config.access_tokens:
            self.logger.warning(f"Token validation failed: token not found")
            return None
        
        token_data = self.config.access_tokens[token]
        
        # Check if token is expired
        if datetime.now() > token_data['expiry']:
            self.logger.warning(f"Token validation failed: token expired")
            return None
        
        self.logger.info(f"Token validated for user: {token_data['user_id']}")
        return token_data
    
    def revoke_token(self, token: str) -> bool:
        """Revoke an access token"""
        if token in self.config.access_tokens:
            del self.config.access_tokens[token]
            self.logger.info(f"Token revoked: {token[:10]}...")
            return True
        return False


class OAuthAuthorizationHandler:
    """Handler for OAuth authorization requests"""
    
    def __init__(self, oauth_server: OAuthServer):
        self.oauth_server = oauth_server
        self.logger = logger
    
    def handle_authorization_request(self, request_params: Dict) -> Dict:
        """
        Process authorization request from client application
        
        Expected parameters:
        - client_id: Client identifier
        - redirect_uri: Redirect URI for callback
        - response_type: 'code' or 'token'
        - scope: Requested permissions
        - state: CSRF protection token
        - user_id: Authenticated user ID (from session)
        """
        client_id = request_params.get('client_id')
        redirect_uri = request_params.get('redirect_uri')
        response_type = request_params.get('response_type', 'code')
        scope = request_params.get('scope', '')
        state = request_params.get('state')
        user_id = request_params.get('user_id')
        
        self.logger.info(f"Processing authorization request: client={client_id}, "
                        f"user={user_id}, redirect={redirect_uri}")
        
        # Validate required parameters
        if not all([client_id, redirect_uri, user_id]):
            self.logger.error("Missing required parameters in authorization request")
            return {
                'success': False,
                'error': 'invalid_request',
                'error_description': 'Missing required parameters'
            }
        
        # Process authorization through OAuth server
        success, redirect_url = self.oauth_server.authorize(
            client_id=client_id,
            redirect_uri=redirect_uri,
            user_id=user_id,
            scope=scope,
            response_type=response_type,
            state=state
        )
        
        if success:
            return {
                'success': True,
                'redirect_url': redirect_url
            }
        else:
            return {
                'success': False,
                'error': 'authorization_failed',
                'error_description': redirect_url
            }
    
    def handle_token_request(self, request_params: Dict) -> Dict:
        """
        Process token exchange request
        
        Expected parameters:
        - grant_type: 'authorization_code'
        - code: Authorization code
        - redirect_uri: Original redirect URI
        - client_id: Client identifier
        - client_secret: Client secret
        """
        grant_type = request_params.get('grant_type')
        code = request_params.get('code')
        redirect_uri = request_params.get('redirect_uri')
        client_id = request_params.get('client_id')
        client_secret = request_params.get('client_secret')
        
        self.logger.info(f"Processing token request: client={client_id}, grant_type={grant_type}")
        
        if grant_type != 'authorization_code':
            return {
                'error': 'unsupported_grant_type',
                'error_description': 'Only authorization_code grant type is supported'
            }
        
        if not all([code, redirect_uri, client_id, client_secret]):
            return {
                'error': 'invalid_request',
                'error_description': 'Missing required parameters'
            }
        
        token_data = self.oauth_server.exchange_code_for_token(
            client_id=client_id,
            client_secret=client_secret,
            code=code,
            redirect_uri=redirect_uri
        )
        
        if token_data:
            return token_data
        else:
            return {
                'error': 'invalid_grant',
                'error_description': 'Invalid authorization code or credentials'
            }


# Main execution and testing
if __name__ == '__main__':
    # Initialize OAuth server
    config = OAuthServerConfig()
    oauth_server = OAuthServer(config)
    handler = OAuthAuthorizationHandler(oauth_server)
    
    print("OAuth 2.0 Authorization Server initialized")
    print(f"Registered clients: {len(config.clients)}")
    print("\nServer is ready to process authorization requests")
