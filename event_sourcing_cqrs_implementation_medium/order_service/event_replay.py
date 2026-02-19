#!/usr/bin/env python3

import sqlite3
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderState:
    order_id: str
    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = "created"
    total: float = 0.0
    payment_received: bool = False


class EventReplayer:
    def __init__(self):
        self.event_handlers = {
            'OrderCreated': self.handle_order_created,
            'ItemAdded': self.handle_item_added,
            'ItemRemoved': self.handle_item_removed,
            'QuantityChanged': self.handle_quantity_changed,
            'OrderCancelled': self.handle_order_cancelled,
            'PaymentProcessed': self.handle_payment_processed,
        }
    
    def replay_events(self, events: List[Dict[str, Any]]) -> OrderState:
        """Replay events to reconstruct order state"""
        if not events:
            return None
        
        # BUG: Not sorting events by timestamp
        # events.sort(key=lambda e: e.get('timestamp', ''))
        
        state = None
        for event in events:
            state = self.apply_event(state, event)
        
        if state:
            state.total = self.calculate_total(state)
        
        return state
    
    def apply_event(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Apply a single event to the order state"""
        event_type = event.get('event_type')
        handler = self.event_handlers.get(event_type)
        
        if handler:
            # BUG: Missing null check for state in some handlers
            state = handler(state, event)
        else:
            logger.warning(f"Unknown event type: {event_type}")
        
        return state
    
    def handle_order_created(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Handle OrderCreated event"""
        order_id = event.get('order_id')
        state = OrderState(order_id=order_id)
        state.status = 'created'
        logger.info(f"Order created: {order_id}")
        return state
    
    def handle_item_added(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Handle ItemAdded event"""
        item_id = event.get('item_id')
        item_name = event.get('item_name')
        price = event.get('price', 0.0)
        quantity = event.get('quantity', 1)
        
        if item_id in state.items:
            state.items[item_id]['quantity'] += quantity
        else:
            state.items[item_id] = {
                'name': item_name,
                'price': price,
                'quantity': quantity
            }
        
        logger.info(f"Item added: {item_id} to order {state.order_id}")
        return state
    
    def handle_item_removed(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Handle ItemRemoved event"""
        item_id = event.get('item_id')
        
        # BUG: Doesn't actually remove items, just logs
        if item_id in state.items:
            logger.info(f"Item removed: {item_id} from order {state.order_id}")
            # Missing: del state.items[item_id]
        
        return state
    
    def handle_quantity_changed(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Handle QuantityChanged event"""
        item_id = event.get('item_id')
        new_quantity = event.get('new_quantity', 0)
        
        if item_id in state.items:
            # BUG: Adds instead of sets quantity
            state.items[item_id]['quantity'] += new_quantity
            logger.info(f"Quantity changed for {item_id} in order {state.order_id}")
        
        return state
    
    def handle_order_cancelled(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Handle OrderCancelled event"""
        # BUG: Doesn't clear items or set total to 0
        # BUG: Status transition is wrong (doesn't override payment status)
        if state.status != 'paid':
            state.status = 'cancelled'
        logger.info(f"Order cancelled: {state.order_id}")
        return state
    
    def handle_payment_processed(self, state: OrderState, event: Dict[str, Any]) -> OrderState:
        """Handle PaymentProcessed event"""
        state.payment_received = True
        state.status = 'paid'
        logger.info(f"Payment processed for order {state.order_id}")
        return state
    
    def calculate_total(self, state: OrderState) -> float:
        """Calculate order total from items"""
        total = 0.0
        
        # BUG: Incorrect iteration logic
        for item_id in state.items:
            item = state.items[item_id]
            # BUG: Sometimes uses wrong calculation or misses items
            total += item['price'] * item.get('quantity', 0)
        
        # Apply tax (10%)
        tax = total * 0.10
        total += tax
        
        # BUG: Discount logic might be applied incorrectly
        # Apply discount if total > 100
        if total > 100:
            discount = total * 0.05
            total -= discount
        
        return round(total, 2)


def get_all_orders() -> List[Dict[str, Any]]:
    """Connect to event store and reconstruct all orders"""
    db_path = '/workspace/event_store.db'
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch all events
        cursor.execute("""
            SELECT order_id, event_type, event_data, timestamp
            FROM events
            ORDER BY timestamp ASC
        """)
        
        rows = cursor.fetchall()
        
        # Group events by order_id
        orders_events = {}
        for row in rows:
            order_id = row['order_id']
            event = {
                'order_id': order_id,
                'event_type': row['event_type'],
                'timestamp': row['timestamp']
            }
            
            # Parse event_data JSON
            if row['event_data']:
                event_data = json.loads(row['event_data'])
                event.update(event_data)
            
            if order_id not in orders_events:
                orders_events[order_id] = []
            orders_events[order_id].append(event)
        
        conn.close()
        
        # Replay events for each order
        replayer = EventReplayer()
        orders_state = []
        
        for order_id, events in orders_events.items():
            state = replayer.replay_events(events)
            if state:
                orders_state.append({
                    'order_id': state.order_id,
                    'total': state.total,
                    'status': state.status,
                    'item_count': sum(item['quantity'] for item in state.items.values())
                })
        
        return orders_state
        
    except Exception as e:
        logger.error(f"Error processing events: {e}")
        return []


def main():
    """Main function to reconstruct all orders and save to file"""
    logger.info("Starting event replay...")
    
    orders = get_all_orders()
    
    # Save to output file
    output_path = '/workspace/orders_state.json'
    with open(output_path, 'w') as f:
        json.dump(orders, f, indent=2)
    
    logger.info(f"Saved {len(orders)} orders to {output_path}")


if __name__ == '__main__':
    main()