import scrapy
import json


class ProductSpider(scrapy.Spider):
    name = 'product_spider'
    
    custom_settings = {
        'FEEDS': {
            '/workspace/output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
    }
    
    def start_requests(self):
        urls = [
            '/workspace/response1.json',
            '/workspace/response2.json',
            '/workspace/response3.json'
        ]
        for url in urls:
            yield scrapy.Request(url, callback=self.parse)
    
    def parse(self, response):
        # Bug 1: Using response.text directly without proper file URL handling
        data = response.text
        
        # Bug 2: Not properly parsing JSON - trying to use it as dict directly
        products = data.get('products', [])
        
        # Bug 3: Wrong field names when extracting
        for product in products:
            yield {
                'id': product['product_id'],  # Wrong key name
                'name': product['product_name'],  # Wrong key name
                'price': product['cost']  # Wrong key name and no type conversion
            }
    
    def closed(self, reason):
        self.logger.info(f'Spider closed: {reason}')