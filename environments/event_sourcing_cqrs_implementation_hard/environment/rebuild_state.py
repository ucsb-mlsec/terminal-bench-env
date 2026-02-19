#!/usr/bin/env python3

import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

def read_events(filepath):
    """Read events from JSONL file line by line."""
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events

def calculate_total(order_state):
    """Calculate the final total for an order."""
    subtotal = Decimal('0')
    
    # Calculate subtotal from all items
    for item_id, item_data in order_state['items'].items():
        item_total = Decimal(str(item_data['price'])) * Decimal(str(item_data['quantity']))
        subtotal += item_total
    
    # Apply discounts - BUG: discounts are being added instead of subtracted
    total = subtotal
    for discount in order_state['discounts']:
        if discount['type'] == 'percentage':
            discount_amount = subtotal * (Decimal(str(discount['value'])) / Decimal('100'))
            total += discount_amount  # BUG: Should be subtracting
        elif discount['type'] == 'fixed':
            total += Decimal(str(discount['value']))  # BUG: Should be subtracting
    
    return float(total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def process_events(events):
    """Process events and rebuild order state."""
    orders = {}
    
    for event in events:
        order_id = event['order_id']
        event_type = event['event_type']
        
        # Initialize order state if it doesn't exist
        if order_id not in orders:
            orders[order_id] = {
                'items': {},
                'discounts': []
            }
        
        order_state = orders[order_id]
        
        # Handle different event types
        if event_type == 'item_added':
            item_id = event['item_id']
            order_state['items'][item_id] = {
                'price': event['price'],
                'quantity': event['quantity']
            }
        
        elif event_type == 'quantity_changed':
            item_id = event['item_id']
            if item_id in order_state['items']:
                order_state['items'][item_id]['quantity'] = event['new_quantity']
        
        elif event_type == 'item_removed':
            item_id = event['item_id']
            if item_id in order_state['items']:
                del order_state['items'][item_id]
        
        elif event_type == 'discount_applied':
            discount = {
                'type': event['discount_type'],
                'value': event['value']
            }
            order_state['discounts'].append(discount)
        
        elif event_type == 'price_adjusted':
            item_id = event['item_id']
            if item_id in order_state['items']:
                order_state['items'][item_id]['price'] = event['new_price']
    
    return orders

def rebuild_state(events_filepath, output_filepath):
    """Main function to rebuild state from events."""
    # Read all events
    events = read_events(events_filepath)
    
    # Process events to rebuild state
    orders = process_events(events)
    
    # Calculate final totals for each order
    final_totals = {}
    for order_id, order_state in orders.items():
        final_totals[order_id] = calculate_total(order_state)
    
    # Save results to output file
    with open(output_filepath, 'w') as f:
        json.dump(final_totals, f, indent=2)
    
    print(f"State rebuilt successfully. Results saved to {output_filepath}")
    print(f"Processed {len(events)} events for {len(final_totals)} orders")

if __name__ == '__main__':
    events_file = '/app/event_store/events.jsonl'
    output_file = '/tmp/rebuilt_state.json'
    
    if not os.path.exists(events_file):
        print(f"Error: Events file not found at {events_file}")
        sys.exit(1)
    
    rebuild_state(events_file, output_file)