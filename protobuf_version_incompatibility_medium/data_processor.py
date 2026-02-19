#!/usr/bin/env python3
"""
Data processor module for handling protobuf messages.
"""

from message_pb2 import User
from transaction_pb2 import Transaction


def process_user_data(user_proto):
    """
    Process and print User protobuf object fields.
    
    Args:
        user_proto: A User protobuf object
    """
    print(f"User ID: {user_proto.id}")
    print(f"Name: {user_proto.name}")
    print(f"Email: {user_proto.email}")
    if hasattr(user_proto, 'age'):
        print(f"Age: {user_proto.age}")
    print("---")


def process_transaction_data(transaction_proto):
    """
    Process and print Transaction protobuf object fields.
    
    Args:
        transaction_proto: A Transaction protobuf object
    """
    print(f"Transaction ID: {transaction_proto.id}")
    print(f"User ID: {transaction_proto.user_id}")
    print(f"Amount: {transaction_proto.amount}")
    if hasattr(transaction_proto, 'timestamp'):
        print(f"Timestamp: {transaction_proto.timestamp}")
    print("---")


def serialize_batch():
    """
    Create and serialize a batch of User objects.
    
    Returns:
        list: A list of serialized User objects as bytes
    """
    users = [
        User(id=1, name="Alice Smith", email="alice@example.com"),
        User(id=2, name="Bob Johnson", email="bob@example.com"),
        User(id=3, name="Charlie Brown", email="charlie@example.com")
    ]
    
    serialized_users = []
    for user in users:
        serialized_users.append(user.SerializeToString())
    
    return serialized_users


if __name__ == "__main__":
    # Test the functions
    print("Testing User processing:")
    user1 = User(id=1, name="Test User", email="test@example.com")
    process_user_data(user1)
    
    print("\nTesting Transaction processing:")
    transaction1 = Transaction(id=101, user_id=1, amount=99.99)
    process_transaction_data(transaction1)
    
    print("\nTesting batch serialization:")
    serialized_batch = serialize_batch()
    print(f"Serialized {len(serialized_batch)} users")
    
    print("\nAll tests completed successfully!")