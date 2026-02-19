#!/usr/bin/env python3

import json
import logging
from datetime import datetime
from validator import validate_customer_info, validate_order_items, validate_payment_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_order_total(items):
    """Calculate the total price for all items in the order."""
    total = 0.0
    for item in items:
        quantity = item.get('quantity', 0)
        price = item.get('price', 0.0)
        total += quantity * price
    return round(total, 2)

def validate_and_process(payload):
    """
    Main validation and processing function.
    Validates customer info, order items, and payment details.
    Returns True if successful, raises exception if validation fails.
    """
    try:
        # Validate customer information
        customer = payload.get('customer', {})
        if not validate_customer_info(customer):
            raise ValueError("Customer information validation failed")
        
        # Validate order items
        items = payload.get('items', [])
        if not validate_order_items(items):
            raise ValueError("Order items validation failed")
        
        # Calculate order total
        calculated_total = calculate_order_total(items)
        expected_total = payload.get('total', 0.0)
        
        # Bug: Using strict equality for floating point comparison
        if calculated_total != expected_total:
            raise ValueError(f"Order total mismatch: expected {expected_total}, calculated {calculated_total}")
        
        # Validate payment information
        payment = payload.get('payment', {})
        if not validate_payment_info(payment):
            raise ValueError("Payment information validation failed")
        
        logger.info(f"Order processed successfully. Total: ${calculated_total}")
        return True
        
    except Exception as e:
        logger.error(f"Order processing failed: {str(e)}")
        raise

def process_order(message_payload):
    """
    Process an order message.
    Returns dict with status and any error message.
    """
    try:
        validate_and_process(message_payload)
        return {"status": "success", "message": "Order processed successfully"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    # Example usage
    sample_order = {
        "customer": {"name": "John Doe", "email": "john@example.com"},
        "items": [{"name": "Widget", "quantity": 2, "price": 10.50}],
        "total": 21.00,
        "payment": {"method": "credit_card"}
    }
    result = process_order(sample_order)
    print(json.dumps(result, indent=2))
```

```python
#!/usr/bin/env python3

import re
import logging

logger = logging.getLogger(__name__)

def validate_customer_info(customer):
    """
    Validate customer information including name, email, and address.
    Returns True if valid, False otherwise.
    """
    if not customer:
        logger.error("Customer information is missing")
        return False
    
    # Validate customer name
    name = customer.get('name', '')
    if not name or len(name) < 2:
        logger.error("Invalid customer name")
        return False
    
    # Validate email
    email = customer.get('email', '')
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        logger.error(f"Invalid email format: {email}")
        return False
    
    # Validate postal code if address is provided
    address = customer.get('address', {})
    if address:
        postal_code = address.get('postal_code', '')
        if postal_code:
            # Bug: Regex doesn't allow hyphens in postal codes
            postal_pattern = r'^[A-Z0-9]{5,10}$'
            if not re.match(postal_pattern, postal_code.upper()):
                logger.error(f"Invalid postal code format: {postal_code}")
                return False
    
    return True

def validate_order_items(items):
    """
    Validate order items list.
    Each item must have name, quantity, and price.
    """
    if not items or not isinstance(items, list):
        logger.error("Items list is missing or invalid")
        return False
    
    # Bug: Should allow empty lists for valid orders, but requires at least 1 item
    if len(items) < 1:
        logger.error("Order must contain at least one item")
        return False
    
    for item in items:
        # Validate item name
        if not item.get('name'):
            logger.error("Item missing name")
            return False
        
        # Validate quantity
        quantity = item.get('quantity', 0)
        if not isinstance(quantity, int) or quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}")
            return False
        
        # Validate price
        price = item.get('price', 0)
        if not isinstance(price, (int, float)) or price < 0:
            logger.error(f"Invalid price: {price}")
            return False
    
    return True

def validate_payment_info(payment):
    """
    Validate payment information.
    Must include valid payment method.
    """
    if not payment:
        logger.error("Payment information is missing")
        return False
    
    method = payment.get('method', '')
    valid_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer']
    
    if method not in valid_methods:
        logger.error(f"Invalid payment method: {method}")
        return False
    
    return True