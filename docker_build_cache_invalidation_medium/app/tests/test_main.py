import pytest
from src.main import app


@pytest.fixture
def client():
    """Create a test client for the application"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert b'healthy' in response.data


def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data


def test_api_data(client):
    """Test the API data endpoint"""
    response = client.get('/api/data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'message' in data
    assert isinstance(data, dict)