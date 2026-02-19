#!/usr/bin/env python3
"""
Validation module for order message processing.
Contains functions to validate various fields in order payloads.
"""

import re


def validate_postal_code(postal_code):
    """
    Validate postal code format.
    
    Args:
        postal_code: String postal code to validate
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    if not postal_code:
        raise ValueError("Postal code cannot be empty")
    
    # BUG: Only accepts 5-digit format, rejects valid 9-digit format with hyphen
    pattern = r'^\d{5}$'
    if not re.match(pattern, postal_code):
        raise ValueError(f"Invalid postal code format: {postal_code}. Must be 5 digits.")
    
    return True


def validate_email(email):
    """
    Validate email address format.
    
    Args:
        email: String email address to validate
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    if not email:
        raise ValueError("Email cannot be empty")
    
    # BUG: Rejects valid emails with plus signs (email+tag@domain.com)
    pattern = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError(f"Invalid email format: {email}")
    
    return True


def validate_phone(phone):
    """
    Validate phone number format.
    
    Args:
        phone: String phone number to validate
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    if not phone:
        raise ValueError("Phone number cannot be empty")
    
    # BUG: Requires exact format with parentheses and hyphen, rejects other valid formats
    pattern = r'^\(\d{3}\) \d{3}-\d{4}$'
    if not re.match(pattern, phone):
        raise ValueError(f"Invalid phone format: {phone}. Must be (XXX) XXX-XXXX")
    
    return True


def validate_quantity(quantity):
    """
    Validate order quantity.
    
    Args:
        quantity: Numeric quantity value
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    # BUG: Incorrectly rejects decimal/float quantities which are valid for items sold by weight
    if not isinstance(quantity, int):
        raise ValueError(f"Invalid quantity type: {type(quantity).__name__}. Must be integer.")
    
    if quantity <= 0:
        raise ValueError(f"Invalid quantity value: {quantity}. Must be positive.")
    
    return True


def validate_order_payload(payload):
    """
    Orchestrates validation of all fields in an order payload.
    
    Args:
        payload: Dictionary containing order data
        
    Returns:
        bool: True if all validations pass
        
    Raises:
        ValueError: If any validation fails with descriptive message
    """
    required_fields = ['customer_email', 'shipping_address', 'phone', 'items']
    
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")
    
    validate_email(payload['customer_email'])
    validate_phone(payload['phone'])
    
    if 'postal_code' in payload.get('shipping_address', {}):
        validate_postal_code(payload['shipping_address']['postal_code'])
    
    for item in payload.get('items', []):
        if 'quantity' in item:
            validate_quantity(item['quantity'])
    
    return True