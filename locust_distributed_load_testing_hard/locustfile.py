from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def view_homepage(self):
        self.client.get("/")
    
    @task(2)
    def view_products(self):
        self.client.get("/products")
    
    @task(2)
    def view_product_detail(self):
        product_id = 1
        self.client.get(f"/products/{product_id}")
    
    @task(1)
    def view_cart(self):
        self.client.get("/cart")
    
    @task(1)
    def add_to_cart(self):
        self.client.post("/cart/add", json={
            "product_id": 1,
            "quantity": 1
        })
    
    @task(1)
    def search(self):
        self.client.get("/search?q=laptop")
    
    def on_start(self):
        # This runs once when a simulated user starts
        self.client.get("/")