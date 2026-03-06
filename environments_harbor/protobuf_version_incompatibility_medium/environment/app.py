#!/usr/bin/env python3

import sys
import message_pb2
import transaction_pb2

def main():
    try:
        # Create a User message
        user = message_pb2.User()
        user.id = 1
        user.name = 'John Doe'
        user.email = 'john@example.com'
        user.age = 30
        user.is_active = True
        
        print(f"Created User: id={user.id}, name={user.name}, email={user.email}, age={user.age}, is_active={user.is_active}")
        
        # Serialize User to binary
        user_binary = user.SerializeToString()
        print(f"User serialized successfully. Binary size: {len(user_binary)} bytes")
        
        # Deserialize User from binary
        user_deserialized = message_pb2.User()
        user_deserialized.ParseFromString(user_binary)
        print(f"User deserialized successfully: id={user_deserialized.id}, name={user_deserialized.name}, email={user_deserialized.email}, age={user_deserialized.age}, is_active={user_deserialized.is_active}")
        
        # Create a Transaction message
        transaction = transaction_pb2.Transaction()
        transaction.transaction_id = 'TXN001'
        transaction.user_id = 1
        transaction.amount = 99.99
        transaction.timestamp = 1234567890
        transaction.status = 'completed'
        
        print(f"\nCreated Transaction: transaction_id={transaction.transaction_id}, user_id={transaction.user_id}, amount={transaction.amount}, timestamp={transaction.timestamp}, status={transaction.status}")
        
        # Serialize Transaction to binary
        transaction_binary = transaction.SerializeToString()
        print(f"Transaction serialized successfully. Binary size: {len(transaction_binary)} bytes")
        
        # Deserialize Transaction from binary
        transaction_deserialized = transaction_pb2.Transaction()
        transaction_deserialized.ParseFromString(transaction_binary)
        print(f"Transaction deserialized successfully: transaction_id={transaction_deserialized.transaction_id}, user_id={transaction_deserialized.user_id}, amount={transaction_deserialized.amount}, timestamp={transaction_deserialized.timestamp}, status={transaction_deserialized.status}")
        
        print("\nAll protobuf operations completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)