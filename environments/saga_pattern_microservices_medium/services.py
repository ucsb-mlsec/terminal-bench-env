#!/usr/bin/env python3

class InsufficientInventoryError(Exception):
    pass

class PaymentDeclinedError(Exception):
    pass

class ShippingFailureError(Exception):
    pass


class InventoryService:
    def __init__(self, initial_stock=None):
        if initial_stock is None:
            initial_stock = {'item1': 100, 'item2': 50, 'item3': 0}
        self.stock = initial_stock.copy()
        self.reservations = {}
    
    def reserve_inventory(self, order_id, item, quantity):
        if item not in self.stock:
            raise InsufficientInventoryError(f"Item {item} not found")
        
        if self.stock[item] < quantity:
            raise InsufficientInventoryError(
                f"Insufficient inventory for {item}. Available: {self.stock[item]}, Requested: {quantity}"
            )
        
        self.stock[item] -= quantity
        self.reservations[order_id] = (item, quantity)
        return True
    
    def release_inventory(self, order_id):
        if order_id in self.reservations:
            item, quantity = self.reservations[order_id]
            self.stock[item] += quantity
            del self.reservations[order_id]
            return True
        return False


class PaymentService:
    def __init__(self, declined_cards=None):
        if declined_cards is None:
            declined_cards = {'CARD-DECLINE-001', 'CARD-DECLINE-002'}
        self.declined_cards = declined_cards
        self.payments = {}
    
    def process_payment(self, order_id, card_number, amount):
        if card_number in self.declined_cards:
            raise PaymentDeclinedError(f"Card {card_number} was declined")
        
        self.payments[order_id] = (card_number, amount)
        return True
    
    def refund_payment(self, order_id):
        if order_id in self.payments:
            del self.payments[order_id]
            return True
        return False


class ShippingService:
    def __init__(self, invalid_addresses=None):
        if invalid_addresses is None:
            invalid_addresses = {'ADDR-FAIL-001', 'ADDR-FAIL-002'}
        self.invalid_addresses = invalid_addresses
        self.shipping_labels = {}
        self.label_counter = 1
    
    def create_shipping_label(self, order_id, address):
        if address in self.invalid_addresses:
            raise ShippingFailureError(f"Cannot ship to address {address}")
        
        label_id = f"LABEL-{self.label_counter:06d}"
        self.label_counter += 1
        self.shipping_labels[order_id] = (address, label_id)
        return label_id
    
    def cancel_shipping_label(self, order_id):
        if order_id in self.shipping_labels:
            del self.shipping_labels[order_id]
            return True
        return False