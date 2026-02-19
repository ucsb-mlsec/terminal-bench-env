#!/usr/bin/env python3
from locust import HttpUser, task, between


class WebsiteUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(3)
    def index_page(self):
        """Load the homepage - most common task"""
        self.client.get("/")
    
    @task(2)
    def api_data(self):
        """Fetch API data"""
        self.client.get("/api/data")
    
    @task(1)
    def login_page(self):
        """Access login page"""
        self.client.get("/login")
    
    @task(1)
    def about_page(self):
        """Load about page"""
        self.client.get("/about")
    
    def on_start(self):
        """Called when a simulated user starts"""
        pass
```

Now, let me create the configuration file:

```ini
[master]
locustfile = /home/user/loadtest/locustfile.py
master = true
headless = true
master-bind-host = 0.0.0.0
master-bind-port = 5557
users = 100
spawn-rate = 10
run-time = 60s
host = http://example.com

[worker]
locustfile = /home/user/loadtest/locustfile.py
worker = true
master-host = localhost
master-port = 5557