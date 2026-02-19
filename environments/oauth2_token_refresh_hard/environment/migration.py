#!/usr/bin/env python3

import requests
import json
import sqlite3
import time

def load_oauth_config():
    with open('oauth_config.json', 'r') as f:
        return json.load(f)

def get_access_token(config):
    """Get initial access token from the OAuth server"""
    response = requests.post(
        'http://localhost:5000/oauth/token',
        json={
            'client_id': config['client_id'],
            'client_secret': config['client_secret']
        }
    )
    response.raise_for_status()
    token_data = response.json()
    return token_data['access_token']

def fetch_customer_page(page_num, access_token):
    """Fetch a single page of customer data"""
    response = requests.get(
        f'http://localhost:5000/customers?page={page_num}',
        headers={'Authorization': f'Bearer {access_token}'}
    )
    response.raise_for_status()
    return response.json()

def insert_customers(customers):
    """Insert customer records into the database"""
    conn = sqlite3.connect('customers.db')
    cursor = conn.cursor()
    
    for customer in customers:
        cursor.execute('''
            INSERT OR IGNORE INTO customers (customer_id, name, email, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            customer['customer_id'],
            customer['name'],
            customer['email'],
            customer['created_at']
        ))
    
    conn.commit()
    conn.close()

def main():
    """Main migration logic"""
    print("Starting migration...")
    
    # Load OAuth configuration
    config = load_oauth_config()
    
    # Get initial access token
    print("Obtaining access token...")
    access_token = get_access_token(config)
    print("Access token obtained")
    
    # Process all 20 pages
    total_records = 0
    for page in range(1, 21):
        print(f"Processing page {page}/20...")
        
        try:
            # Fetch customer data for this page
            page_data = fetch_customer_page(page, access_token)
            customers = page_data['customers']
            
            # Insert customers into database
            insert_customers(customers)
            total_records += len(customers)
            
            print(f"  Inserted {len(customers)} records from page {page}")
            
            # Simulate network delay
            time.sleep(1)
            
        except requests.exceptions.HTTPError as e:
            print(f"ERROR: Failed to process page {page}: {e}")
            raise
    
    print(f"\nMigration complete! Total records: {total_records}")

if __name__ == '__main__':
    main()