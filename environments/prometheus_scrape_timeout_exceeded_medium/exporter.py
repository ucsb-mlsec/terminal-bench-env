#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer
import time

# Custom metrics exporter with intentional performance issue
# This exporter simulates slow data collection that causes scrape timeouts

class MetricsHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            
            # Simulate slow metric collection process
            # This represents gathering data from multiple sources
            metrics = self.collect_metrics()
            
            self.wfile.write(metrics.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def collect_metrics(self):
        """
        Collect metrics from various sources.
        WARNING: This function has a performance issue!
        """
        
        # Simulate database query - takes 8 seconds
        time.sleep(8)
        db_value = 42
        
        # Simulate API call to external service - takes 7 seconds
        time.sleep(7)
        api_value = 100
        
        # Simulate file I/O operations - takes 5 seconds
        time.sleep(5)
        file_value = 75
        
        # Total delay: 8 + 7 + 5 = 20 seconds
        # This is too slow for typical Prometheus scrape intervals!
        
        # Build Prometheus metrics format
        metrics = []
        metrics.append('# HELP custom_db_metric Value from database query')
        metrics.append('# TYPE custom_db_metric gauge')
        metrics.append(f'custom_db_metric {db_value}')
        metrics.append('')
        metrics.append('# HELP custom_api_metric Value from external API')
        metrics.append('# TYPE custom_api_metric gauge')
        metrics.append(f'custom_api_metric {api_value}')
        metrics.append('')
        metrics.append('# HELP custom_file_metric Value from file operations')
        metrics.append('# TYPE custom_file_metric gauge')
        metrics.append(f'custom_file_metric {file_value}')
        metrics.append('')
        
        return '\n'.join(metrics)
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MetricsHandler)
    print(f'Starting metrics exporter on port {port}...')
    print(f'Metrics available at http://localhost:{port}/metrics')
    print('WARNING: This exporter has intentional delays for testing purposes')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()