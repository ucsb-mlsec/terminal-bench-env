#!/usr/bin/env python3

import requests
import json
import time
import uuid
import sys

# API base URL
API_BASE_URL = 'http://localhost:5000'

def make_payment(amount, currency, customer_id, idempotency_key=None):
    """
    Make a payment request to the API.
    
    Args:
        amount: Payment amount
        currency: Currency code (e.g., 'USD')
        customer_id: Customer identifier
        idempotency_key: Optional idempotency key for duplicate prevention
    
    Returns:
        Response object from the API
    """
    url = f'{API_BASE_URL}/api/payment'
    
    payload = {
        'amount': amount,
        'currency': currency,
        'customer_id': customer_id
    }
    
    headers = {'Content-Type': 'application/json'}
    
    # Add idempotency key to headers if provided
    if idempotency_key:
        headers['Idempotency-Key'] = idempotency_key
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        return response
    except requests.exceptions.Timeout:
        print("Request timed out!")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error!")
        return None
    except Exception as e:
        print(f"Error making payment: {e}")
        return None

def get_transaction(transaction_id):
    """
    Retrieve transaction details by ID.
    
    Args:
        transaction_id: The transaction ID to retrieve
    
    Returns:
        Transaction details as JSON
    """
    url = f'{API_BASE_URL}/api/transaction/{transaction_id}'
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error retrieving transaction: {e}")
        return None

def print_payment_response(response, label="Payment"):
    """
    Print formatted payment response.
    
    Args:
        response: Response object from payment API
        label: Label to identify this response
    """
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    if response is None:
        print("No response received (timeout or connection error)")
        return
    
    print(f"Status Code: {response.status_code}")
    
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Response: {response.text}")
    
    print(f"{'='*60}\n")

if __name__ == '__main__':
    print("Payment API Client - Demonstrating Duplicate Charge Issue")
    print("="*60)
    
    # Test data
    amount = 100.00
    currency = 'USD'
    customer_id = 'cust_12345'
    
    print(f"\nAttempting to charge customer {customer_id} for ${amount} {currency}")
    print("\n--- SCENARIO: Network timeout causes client to retry ---")
    print("This demonstrates the duplicate charge problem when retries occur.\n")
    
    # First payment attempt
    print("1. Making initial payment request...")
    response1 = make_payment(amount, currency, customer_id)
    print_payment_response(response1, "First Payment Attempt")
    
    # Simulate network issue - client thinks request failed and retries
    print("2. Network timeout occurred (simulated)...")
    print("   Client thinks request failed and retries...\n")
    time.sleep(1)
    
    # Retry - this creates a duplicate charge!
    print("3. Retrying payment request...")
    response2 = make_payment(amount, currency, customer_id)
    print_payment_response(response2, "Second Payment Attempt (RETRY)")
    
    # Show the problem
    print("\n" + "!"*60)
    print("PROBLEM IDENTIFIED:")
    print("!"*60)
    
    if response1 and response2:
        try:
            data1 = response1.json()
            data2 = response2.json()
            
            if response1.status_code == 200 and response2.status_code == 200:
                print(f"First charge:  Transaction ID = {data1.get('transaction_id')}")
                print(f"Second charge: Transaction ID = {data2.get('transaction_id')}")
                print(f"\nCustomer was charged TWICE for the same purchase!")
                print(f"Total charged: ${amount * 2} {currency}")
                print(f"Expected charge: ${amount} {currency}")
                print(f"\nThis is a DUPLICATE CHARGE BUG!")
        except:
            pass
    
    print("\n" + "!"*60)
    print("\nSOLUTION NEEDED:")
    print("- Implement idempotency keys to detect duplicate requests")
    print("- Return cached response for retry requests")
    print("- Prevent duplicate charges even when clients retry")
    print("="*60)
