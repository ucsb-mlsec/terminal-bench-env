#!/usr/bin/env python3

import json
import requests
import datetime
import time
import sys
import os
from pathlib import Path

class SyncService:
    def __init__(self):
        self.config_path = '/opt/sync-service/auth_config.json'
        self.log_path = '/var/log/sync-service/sync_service_error.log'
        self.config = self.load_config()
        
    def load_config(self):
        """Load OAuth2 configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.log_error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            self.log_error(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def save_config(self):
        """Save configuration back to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.log_error(f"Failed to save configuration: {e}")
    
    def log_error(self, message):
        """Log error messages to file"""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            timestamp = datetime.datetime.now().isoformat()
            with open(self.log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}", file=sys.stderr)
    
    def log_info(self, message):
        """Log informational messages"""
        timestamp = datetime.datetime.now().isoformat()
        log_path = self.log_path.replace('_error.log', '.log')
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}", file=sys.stderr)
    
    def is_token_expired(self):
        """Check if the current access token is expired"""
        try:
            expires_at = self.config.get('token_expires_at')
            if not expires_at:
                self.log_info("No token expiration time found, treating as expired")
                return True
            
            expiry_time = datetime.datetime.fromisoformat(expires_at)
            current_time = datetime.datetime.now()
            
            # Add 5 minute buffer to avoid edge cases
            buffer = datetime.timedelta(minutes=5)
            is_expired = current_time >= (expiry_time - buffer)
            
            if is_expired:
                self.log_info(f"Access token expired at {expires_at}")
            else:
                self.log_info(f"Access token still valid until {expires_at}")
            
            return is_expired
            
        except Exception as e:
            self.log_error(f"Error checking token expiration: {e}")
            return True
    
    def refresh_access_token(self):
        """Use refresh token to obtain a new access token"""
        try:
            self.log_info("Attempting to refresh access token...")
            
            token_endpoint = self.config.get('token_endpoint')
            refresh_token = self.config.get('refresh_token')
            client_id = self.config.get('client_id')
            client_secret = self.config.get('client_secret')
            
            if not all([token_endpoint, refresh_token, client_id, client_secret]):
                self.log_error("Missing required OAuth2 configuration parameters")
                return False
            
            payload = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': client_id,
                'client_secret': client_secret
            }
            
            response = requests.post(
                token_endpoint,
                data=payload,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                
                # Update access token
                self.config['access_token'] = token_data.get('access_token')
                
                # Update refresh token if new one provided
                if 'refresh_token' in token_data:
                    self.config['refresh_token'] = token_data.get('refresh_token')
                    self.log_info("Refresh token was rotated by provider")
                
                # Calculate and save expiration time
                expires_in = token_data.get('expires_in', 3600)
                expiry_time = datetime.datetime.now() + datetime.timedelta(seconds=expires_in)
                self.config['token_expires_at'] = expiry_time.isoformat()
                self.config['last_token_refresh'] = datetime.datetime.now().isoformat()
                
                self.save_config()
                self.log_info("Successfully refreshed access token")
                return True
            
            elif response.status_code == 400:
                error_data = response.json()
                error_type = error_data.get('error', 'unknown')
                error_description = error_data.get('error_description', 'No description')
                
                if error_type == 'invalid_grant':
                    self.log_error(f"invalid_grant error: {error_description}")
                    self.log_error("Refresh token is no longer valid. Manual re-authentication required.")
                    
                    # Perform diagnosis
                    self.diagnose_token_failure()
                else:
                    self.log_error(f"Token refresh failed with error: {error_type} - {error_description}")
                
                return False
            
            else:
                self.log_error(f"Token refresh failed with status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_error(f"Network error during token refresh: {e}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error during token refresh: {e}")
            return False
    
    def diagnose_token_failure(self):
        """Diagnose why token refresh is failing"""
        try:
            self.log_info("Starting OAuth2 token failure diagnosis...")
            
            # Calculate days since service started
            service_start = self.config.get('service_start_date')
            last_refresh = self.config.get('last_token_refresh')
            
            days_since_last_refresh = 0
            root_cause = "Unknown"
            corrective_action = "Unknown"
            
            if last_refresh:
                last_refresh_date = datetime.datetime.fromisoformat(last_refresh)
                days_since_last_refresh = (datetime.datetime.now() - last_refresh_date).days
                self.log_info(f"Days since last successful refresh: {days_since_last_refresh}")
            elif service_start:
                start_date = datetime.datetime.fromisoformat(service_start)
                days_since_last_refresh = (datetime.datetime.now() - start_date).days
                self.log_info(f"Days since service start (no refresh recorded): {days_since_last_refresh}")
            
            # Determine root cause
            if days_since_last_refresh > 90:
                root_cause = "Refresh token expired after 90 days without being used"
                corrective_action = "Manual re-authentication required to obtain new access and refresh tokens"
            elif days_since_last_refresh >= 0:
                root_cause = "Refresh token has exceeded its 90-day lifetime and is no longer valid"
                corrective_action = "Initiate OAuth2 authorization flow to obtain new tokens with proper refresh cycle"
            
            # Check if access token was never expiring (thus refresh never called)
            token_expires_at = self.config.get('token_expires_at')
            if token_expires_at:
                expiry = datetime.datetime.fromisoformat(token_expires_at)
                age = (datetime.datetime.now() - expiry).days
                if age > 90:
                    root_cause = "Access token not refreshed within 90-day window, causing refresh token to expire"
                    corrective_action = "Re-authenticate via OAuth2 flow and ensure token refresh occurs before 90-day expiration"
            
            diagnosis = {
                "root_cause": root_cause,
                "days_since_last_refresh": days_since_last_refresh,
                "corrective_action": corrective_action
            }
            
            # Save diagnosis
            diagnosis_path = '/tmp/oauth_diagnosis.json'
            os.makedirs(os.path.dirname(diagnosis_path), exist_ok=True)
            with open(diagnosis_path, 'w') as f:
                json.dump(diagnosis, f, indent=2)
            
            self.log_info(f"Diagnosis saved to {diagnosis_path}")
            self.log_info(f"Root cause: {root_cause}")
            self.log_info(f"Corrective action: {corrective_action}")
            
        except Exception as e:
            self.log_error(f"Error during diagnosis: {e}")
    
    def perform_sync(self):
        """Perform data synchronization using valid access token"""
        try:
            self.log_info("Starting data synchronization...")
            
            api_endpoint = self.config.get('api_endpoint')
            access_token = self.config.get('access_token')
            
            if not api_endpoint or not access_token:
                self.log_error("Missing API endpoint or access token")
                return False
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                api_endpoint,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                self.log_info("Data synchronization completed successfully")
                return True
            elif response.status_code == 401:
                self.log_error("Access token rejected by API (401 Unauthorized)")
                return False
            else:
                self.log_error(f"API request failed with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_error(f"Network error during synchronization: {e}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error during synchronization: {e}")
            return False
    
    def run(self):
        """Main execution method"""
        try:
            self.log_info("=" * 60)
            self.log_info("Sync service starting execution")
            
            # Check if access token is expired
            if self.is_token_expired():
                self.log_info("Access token expired, attempting refresh...")
                
                if not self.refresh_access_token():
                    self.log_error("Failed to refresh access token, aborting sync")
                    return False
            
            # Perform synchronization
            success = self.perform_sync()
            
            if success:
                self.log_info("Sync service completed successfully")
            else:
                self.log_error("Sync service completed with errors")
            
            self.log_info("=" * 60)
            return success
            
        except Exception as e:
            self.log_error(f"Critical error in run method: {e}")
            return False


def main():
    """Main entry point"""
    try:
        service = SyncService()
        success = service.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()