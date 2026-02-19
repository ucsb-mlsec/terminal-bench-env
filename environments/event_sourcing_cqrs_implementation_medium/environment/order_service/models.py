#!/usr/bin/env python3

from typing import Dict, Any


class OrderState:
    """Represents the current state of an order after replaying events."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.items: Dict[str, Dict[str, Any]] = {}  # item_id -> {name, price, quantity}
        self.status = 'created'
        self.total = 0.0
        self.payment_received = False
        self.discount = 0.0
        self.tax_rate = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order state to dictionary for JSON serialization."""
        item_count = sum(item['quantity'] for item in self.items.values())
        return {
            'order_id': self.order_id,
            'total': round(self.total, 2),
            'status': self.status,
            'item_count': item_count
        }
    
    def calculate_total(self):
        """Calculate the total price including discounts and taxes."""
        subtotal = sum(item['price'] * item['quantity'] for item in self.items.values())
        discounted = subtotal - self.discount
        self.total = discounted * (1 + self.tax_rate)


class Event:
    """Represents an event from the event store."""
    
    def __init__(self, id: int, order_id: str, event_type: str, timestamp: str, payload: Dict[str, Any]):
        self.id = id
        self.order_id = order_id
        self.event_type = event_type
        self.timestamp = timestamp
        self.payload = payload
    
    @classmethod
    def from_db_row(cls, row: tuple) -> 'Event':
        """Create an Event instance from a database row."""
        import json
        
        # Assuming row format: (id, order_id, event_type, timestamp, payload_json)
        id, order_id, event_type, timestamp, payload_json = row
        payload = json.loads(payload_json) if payload_json else {}
        
        return cls(id, order_id, event_type, timestamp, payload)