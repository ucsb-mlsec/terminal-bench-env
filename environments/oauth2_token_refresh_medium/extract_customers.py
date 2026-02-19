#!/usr/bin/env python3

import json
import time
import sys
import os
import subprocess

def read_token():
    """Read the current token from the token file."""
    token_file = '/data/auth/token.json'
    with open(token_file, 'r') as f:
        token_data = json.load(f)
    return token_data['access_token'], token_data['expires_at']

def is_token_expired(expires_at):
    """Check if the token is expired."""
    return time.time() >= expires_at

def refresh_token():
    """Refresh the token by calling the refresh script."""
    print("Token expired, refreshing...")
    subprocess.run(['/data/auth/refresh_token.sh'], check=True)
    print("Token refreshed successfully")

def authenticate():
    """Get a valid token, refreshing if necessary."""
    access_token, expires_at = read_token()
    
    if is_token_expired(expires_at):
        refresh_token()
        access_token, expires_at = read_token()
    
    return access_token, expires_at

def extract_customers():
    """Extract customer data from all pages with token refresh handling."""
    all_customers = []
    pages_processed = 0
    processed_pages = set()
    
    # Track which pages we've already processed to avoid duplicates
    progress_file = '/tmp/extraction_progress.json'
    
    # Load previous progress if it exists
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            processed_pages = set(progress.get('processed_pages', []))
            all_customers = progress.get('customers', [])
            print(f"Resuming from previous progress: {len(processed_pages)} pages already processed")
    
    # Get initial token
    access_token, expires_at = authenticate()
    
    # Process all pages
    for page_num in range(1, 101):
        page_file = f'/data/customer_export/page_{page_num:03d}.json'
        
        # Skip if already processed
        if page_num in processed_pages:
            print(f"Skipping already processed page {page_num}")
            continue
        
        # Check token before processing each page
        if is_token_expired(expires_at):
            refresh_token()
            access_token, expires_at = read_token()
        
        # Read customer data from page
        if os.path.exists(page_file):
            with open(page_file, 'r') as f:
                page_data = json.load(f)
                customers = page_data.get('customers', [])
                all_customers.extend(customers)
                processed_pages.add(page_num)
                pages_processed += 1
                
                print(f"Processed page {page_num}: {len(customers)} customers (Total: {len(all_customers)})")
                
                # Save progress after each page
                with open(progress_file, 'w') as pf:
                    json.dump({
                        'processed_pages': list(processed_pages),
                        'customers': all_customers
                    }, pf)
        else:
            print(f"Warning: Page file {page_file} not found")
        
        # Simulate processing time
        time.sleep(0.1)
    
    # Save final output
    output_file = '/workspace/customers_output.json'
    with open(output_file, 'w') as f:
        json.dump(all_customers, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Total records: {len(all_customers)}")
    print(f"Pages processed: {pages_processed}")
    print(f"Output saved to: {output_file}")
    
    # Write summary file
    summary_file = '/workspace/extraction_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"total_records={len(all_customers)}\n")
        f.write(f"pages_processed={len(processed_pages)}\n")
        f.write(f"completed={'yes' if len(processed_pages) == 100 else 'no'}\n")
    
    print(f"Summary written to: {summary_file}")
    
    # Clean up progress file on successful completion
    if len(processed_pages) == 100 and os.path.exists(progress_file):
        os.remove(progress_file)
    
    return len(all_customers), len(processed_pages)

if __name__ == '__main__':
    try:
        total_records, pages_processed = extract_customers()
        sys.exit(0)
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)