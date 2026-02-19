import time

def auth_backend():
    """Simulates a fast, reliable authentication service"""
    time.sleep(0.05)
    return {"authenticated": True, "user_id": 12345}

def data_backend():
    """Simulates a fast, reliable data retrieval service"""
    time.sleep(0.1)
    return {"data": ["item1", "item2", "item3"], "count": 3}

def reports_backend():
    """Simulates a problematic reports service that times out"""
    time.sleep(10)
    return {"report": "analytics", "rows": 1000}