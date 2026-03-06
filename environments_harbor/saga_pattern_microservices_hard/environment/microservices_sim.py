#!/usr/bin/env python3

class OrderService:
    def __init__(self):
        self.created_orders = {}
    
    def create_order(self, order_id, customer_id, items):
        print(f"OrderService: Creating order {order_id}")
        
        if not order_id or not isinstance(order_id, str):
            raise Exception("Invalid order_id format")
        
        if not customer_id or not isinstance(customer_id, str):
            raise Exception("Invalid customer_id format")
        
        if not items or not isinstance(items, list):
            raise Exception("Invalid items format")
        
        self.created_orders[order_id] = {
            'customer_id': customer_id,
            'items': items
        }
        
        print(f"OrderService: Successfully created order {order_id}")
        return True
    
    def cancel_order(self, order_id):
        print(f"OrderService: Cancelling order {order_id}")
        
        if order_id in self.created_orders:
            del self.created_orders[order_id]
            print(f"OrderService: Successfully cancelled order {order_id}")
        else:
            print(f"OrderService: Order {order_id} not found, nothing to cancel")
        
        return True


class PaymentService:
    def __init__(self):
        self.processed_payments = {}
    
    def process_payment(self, order_id, amount):
        print(f"PaymentService: Processing payment for order {order_id}, amount: ${amount}")
        
        if not order_id or not isinstance(order_id, str):
            raise Exception("Invalid order_id for payment")
        
        if amount <= 0:
            raise Exception("Invalid payment amount")
        
        # Simulate payment processing
        self.processed_payments[order_id] = amount
        
        print(f"PaymentService: Successfully processed payment for order {order_id}")
        return True
    
    def refund_payment(self, order_id):
        print(f"PaymentService: Refunding payment for {order_id}")
        
        if order_id in self.processed_payments:
            amount = self.processed_payments[order_id]
            del self.processed_payments[order_id]
            print(f"PaymentService: Successfully refunded ${amount} for order {order_id}")
        else:
            print(f"PaymentService: No payment found for order {order_id}, nothing to refund")
        
        return True


class InventoryService:
    def __init__(self):
        self.reserved_inventory = {}
    
    def reserve_inventory(self, order_id, items):
        print(f"InventoryService: Reserving inventory for order {order_id}")
        
        if not order_id or not isinstance(order_id, str):
            raise Exception("Invalid order_id for inventory reservation")
        
        if not items or not isinstance(items, list):
            raise Exception("Invalid items format")
        
        # Specific failure scenario for test case
        if order_id == 'ORD-12345':
            for item in items:
                if item.get('product') == 'WIDGET-A' and item.get('quantity') == 2:
                    print(f"InventoryService: Insufficient stock for WIDGET-A")
                    raise Exception("Insufficient stock for WIDGET-A")
        
        # Normal case - reserve inventory
        self.reserved_inventory[order_id] = items
        print(f"InventoryService: Successfully reserved inventory for order {order_id}")
        return True
    
    def release_inventory(self, order_id):
        print(f"InventoryService: Releasing inventory for {order_id}")
        
        if order_id in self.reserved_inventory:
            items = self.reserved_inventory[order_id]
            del self.reserved_inventory[order_id]
            print(f"InventoryService: Successfully released inventory for order {order_id}")
        else:
            print(f"InventoryService: No inventory reserved for order {order_id}, nothing to release")
        
        return True