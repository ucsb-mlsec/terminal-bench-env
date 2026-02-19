import scrapy
import json
import os


class FeedSpider(scrapy.Spider):
    name = 'feed_spider'
    start_urls = [
        'file:///workspace/feeds/partner_a.json',
        'file:///workspace/feeds/partner_b.json',
        'file:///workspace/feeds/partner_c.json',
    ]
    
    custom_settings = {
        'FEED_FORMAT': 'json',
        'FEED_URI': '/workspace/output/products.json',
        'FEED_EXPORT_INDENT': 2,
    }
    
    def __init__(self, *args, **kwargs):
        super(FeedSpider, self).__init__(*args, **kwargs)
        self.all_products = []
    
    def parse(self, response):
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON from {response.url}")
            return
        
        products = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # partner_a.json - flat array
            products = data
        elif isinstance(data, dict):
            if 'data' in data:
                # partner_b.json - nested under "data" key
                products = data.get('data', [])
            elif 'results' in data:
                # partner_c.json - nested under "results" key
                products = data.get('results', [])
        
        for product in products:
            if not isinstance(product, dict):
                continue
            
            # Extract ID with various field names
            product_id = (
                product.get('id') or 
                product.get('product_id') or 
                product.get('sku')
            )
            
            # Extract name with various field names
            product_name = (
                product.get('name') or 
                product.get('product_name') or 
                product.get('title')
            )
            
            # Skip if both ID and name are missing
            if not product_id and not product_name:
                continue
            
            # Extract price with various field names, handle null
            product_price = (
                product.get('price') if product.get('price') is not None else
                product.get('unit_price') if product.get('unit_price') is not None else
                None
            )
            
            # Store product
            self.all_products.append({
                'id': str(product_id) if product_id else '',
                'name': str(product_name) if product_name else '',
                'price': product_price
            })
    
    def closed(self, reason):
        # Sort by ID and write to output file
        sorted_products = sorted(self.all_products, key=lambda x: x['id'])
        
        os.makedirs('/workspace/output', exist_ok=True)
        with open('/workspace/output/products.json', 'w') as f:
            json.dump(sorted_products, f, indent=2)
