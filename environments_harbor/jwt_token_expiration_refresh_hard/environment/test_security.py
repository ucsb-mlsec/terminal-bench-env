#!/usr/bin/env python3

import pytest
import json
import time
from auth_api import app, init_db

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    app.config['DATABASE'] = '/app/auth.db'
    
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

def test_login_success(client):
    """Test that login with valid credentials returns tokens."""
    response = client.post('/login',
                          data=json.dumps({
                              'username': 'testuser',
                              'password': 'testpass123'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'access_token' in data
    assert 'refresh_token' in data
    assert data['access_token'] is not None
    assert data['refresh_token'] is not None
    assert len(data['access_token']) > 0
    assert len(data['refresh_token']) > 0

def test_refresh_token_can_only_be_used_once(client):
    """Test that a refresh token can only be used once."""
    # Login to get initial tokens
    login_response = client.post('/login',
                                 data=json.dumps({
                                     'username': 'testuser',
                                     'password': 'testpass123'
                                 }),
                                 content_type='application/json')
    
    assert login_response.status_code == 200
    login_data = json.loads(login_response.data)
    refresh_token = login_data['refresh_token']
    
    # Use the refresh token once - should succeed
    first_refresh = client.post('/refresh',
                               data=json.dumps({
                                   'refresh_token': refresh_token
                               }),
                               content_type='application/json')
    
    assert first_refresh.status_code == 200
    first_data = json.loads(first_refresh.data)
    assert 'access_token' in first_data
    assert 'refresh_token' in first_data
    
    # Try to use the SAME refresh token again - should fail
    second_refresh = client.post('/refresh',
                                data=json.dumps({
                                    'refresh_token': refresh_token
                                }),
                                content_type='application/json')
    
    assert second_refresh.status_code == 401
    second_data = json.loads(second_refresh.data)
    assert 'error' in second_data or 'message' in second_data

def test_refresh_token_rotation(client):
    """Test that refresh tokens are properly rotated."""
    # Login to get initial tokens
    login_response = client.post('/login',
                                 data=json.dumps({
                                     'username': 'testuser',
                                     'password': 'testpass123'
                                 }),
                                 content_type='application/json')
    
    login_data = json.loads(login_response.data)
    token1 = login_data['refresh_token']
    
    # First refresh
    refresh1 = client.post('/refresh',
                          data=json.dumps({
                              'refresh_token': token1
                          }),
                          content_type='application/json')
    
    assert refresh1.status_code == 200
    refresh1_data = json.loads(refresh1.data)
    token2 = refresh1_data['refresh_token']
    assert token2 != token1
    
    # Second refresh with token2
    refresh2 = client.post('/refresh',
                          data=json.dumps({
                              'refresh_token': token2
                          }),
                          content_type='application/json')
    
    assert refresh2.status_code == 200
    refresh2_data = json.loads(refresh2.data)
    token3 = refresh2_data['refresh_token']
    assert token3 != token2
    assert token3 != token1
    
    # Try to reuse token2 - should fail
    reuse_attempt = client.post('/refresh',
                               data=json.dumps({
                                   'refresh_token': token2
                               }),
                               content_type='application/json')
    
    assert reuse_attempt.status_code == 401

def test_token_family_invalidation_on_reuse(client):
    """Test that reusing an old token invalidates the entire token family."""
    # Login to get initial tokens
    login_response = client.post('/login',
                                 data=json.dumps({
                                     'username': 'testuser',
                                     'password': 'testpass123'
                                 }),
                                 content_type='application/json')
    
    login_data = json.loads(login_response.data)
    token1 = login_data['refresh_token']
    
    # Refresh with token1 to get token2
    refresh1 = client.post('/refresh',
                          data=json.dumps({
                              'refresh_token': token1
                          }),
                          content_type='application/json')
    
    assert refresh1.status_code == 200
    token2 = json.loads(refresh1.data)['refresh_token']
    
    # Refresh with token2 to get token3
    refresh2 = client.post('/refresh',
                          data=json.dumps({
                              'refresh_token': token2
                          }),
                          content_type='application/json')
    
    assert refresh2.status_code == 200
    token3 = json.loads(refresh2.data)['refresh_token']
    
    # Try to reuse token1 (earlier in chain) - should trigger security breach
    reuse_old = client.post('/refresh',
                           data=json.dumps({
                               'refresh_token': token1
                           }),
                           content_type='application/json')
    
    assert reuse_old.status_code == 401
    
    # Now try to use token3 (which was valid) - should also fail due to family invalidation
    use_token3 = client.post('/refresh',
                            data=json.dumps({
                                'refresh_token': token3
                            }),
                            content_type='application/json')
    
    assert use_token3.status_code == 401

def test_access_token_validation(client):
    """Test that access tokens are properly formatted and valid."""
    # Login to get tokens
    login_response = client.post('/login',
                                 data=json.dumps({
                                     'username': 'testuser',
                                     'password': 'testpass123'
                                 }),
                                 content_type='application/json')
    
    assert login_response.status_code == 200
    login_data = json.loads(login_response.data)
    access_token = login_data['access_token']
    
    # Verify token format (JWT has 3 parts separated by dots)
    assert access_token.count('.') == 2
    parts = access_token.split('.')
    assert len(parts) == 3
    assert all(len(part) > 0 for part in parts)

def test_invalid_credentials(client):
    """Test that login with invalid credentials fails."""
    response = client.post('/login',
                          data=json.dumps({
                              'username': 'testuser',
                              'password': 'wrongpassword'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert 'error' in data or 'message' in data

def test_invalid_refresh_token(client):
    """Test that refresh with invalid token fails."""
    # Try with a completely fake token
    response = client.post('/refresh',
                          data=json.dumps({
                              'refresh_token': 'fake.invalid.token'
                          }),
                          content_type='application/json')
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert 'error' in data or 'message' in data
    
    # Try with a malformed token
    response2 = client.post('/refresh',
                           data=json.dumps({
                               'refresh_token': 'notevenatoken'
                           }),
                           content_type='application/json')
    
    assert response2.status_code == 401