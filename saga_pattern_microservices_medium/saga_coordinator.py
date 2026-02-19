#!/usr/bin/env python3

from services import InventoryService, PaymentService, ShippingService

class SagaCoordinator:
    def __init__(self, inventory_service, payment_service, shipping_service):
        self.inventory_service = inventory_service
        self.payment_service = payment_service
        self.shipping_service = shipping_service
    
    def process_order(self, order):
        """
        Process an order through all services with proper saga pattern implementation.
        Includes compensation logic to rollback completed steps on failure.
        """
        order_id = order['order_id']
        item = order['item']
        quantity = order['quantity']
        card_number = order['card_number']
        amount = order['amount']
        address = order['address']
        
        completed_steps = []
        
        try:
            # Step 1: Reserve inventory
            self.inventory_service.reserve(order_id, item, quantity)
            completed_steps.append('inventory')
            
            # Step 2: Process payment
            self.payment_service.charge(order_id, card_number, amount)
            completed_steps.append('payment')
            
            # Step 3: Create shipping label
            self.shipping_service.create_label(order_id, address)
            completed_steps.append('shipping')
            
            return True
            
        except Exception as e:
            # Compensation: rollback completed steps in reverse order
            self._compensate(order_id, completed_steps)
            return False
    
    def _compensate(self, order_id, completed_steps):
        """
        Compensate (rollback) completed steps in reverse order.
        """
        # Process compensations in reverse order
        for step in reversed(completed_steps):
            try:
                if step == 'shipping':
                    self.shipping_service.cancel_label(order_id)
                elif step == 'payment':
                    self.payment_service.refund(order_id)
                elif step == 'inventory':
                    self.inventory_service.release(order_id)
            except Exception as e:
                # Log compensation failure but continue with other compensations
                print(f"Compensation failed for {step} on order {order_id}: {e}")
    
    def verify_consistency(self, order_id):
        """
        Verify that an order is in a consistent state.
        Returns tuple: (is_consistent, outcome)
        - is_consistent: True if all services are in sync (all completed or all rolled back)
        - outcome: 'completed' or 'compensated'
        """
        has_inventory = self.inventory_service.has_reservation(order_id)
        has_payment = self.payment_service.has_charge(order_id)
        has_shipping = self.shipping_service.has_label(order_id)
        
        # Check if all completed
        if has_inventory and has_payment and has_shipping:
            return (True, 'completed')
        
        # Check if all rolled back
        if not has_inventory and not has_payment and not has_shipping:
            return (True, 'compensated')
        
        # Inconsistent state
        return (False, 'inconsistent')


if __name__ == '__main__':
    import json
    
    # Initialize services
    inventory_service = InventoryService()
    payment_service = PaymentService()
    shipping_service = ShippingService()
    
    # Create coordinator
    coordinator = SagaCoordinator(inventory_service, payment_service, shipping_service)
    
    # Load test orders
    with open('/workspace/test_orders.json', 'r') as f:
        test_orders = json.load(f)
    
    # Process all orders and verify consistency
    results = []
    
    for order in test_orders:
        order_id = order['order_id']
        success = coordinator.process_order(order)
        is_consistent, outcome = coordinator.verify_consistency(order_id)
        
        consistent_str = 'yes' if is_consistent else 'no'
        results.append(f"{order_id}:{outcome}:{consistent_str}")
    
    # Save verification results
    with open('/workspace/saga_verification.txt', 'w') as f:
        f.write('\n'.join(results) + '\n')
    
    print("Saga coordinator processing complete. Results saved to saga_verification.txt")