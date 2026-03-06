#!/usr/bin/env python3
"""
Data model classes for the application.
"""


class User:
    """Represents a user in the system."""
    
    def __init__(self, name, email):
        """
        Initialize a User instance.
        
        Args:
            name: The user's name
            email: The user's email address
        """
        self.name = name
        self.email = email
    
    def get_info(self):
        """
        Get user information as a string.
        
        Returns:
            A formatted string with user details
        """
        return f"User: {self.name}, Email: {self.email}"


class Product:
    """Represents a product in the system."""
    
    def __init__(self, id, name, price):
        """
        Initialize a Product instance.
        
        Args:
            id: The product's unique identifier
            name: The product's name
            price: The product's price
        """
        self.id = id
        self.name = name
        self.price = price
    
    def get_price(self):
        """
        Get the product's price.
        
        Returns:
            The price of the product
        """
        return self.price


def create_user(name, email):
    """
    Factory function to create a User instance.
    
    Args:
        name: The user's name
        email: The user's email address
    
    Returns:
        A new User instance
    """
    return User(name, email)