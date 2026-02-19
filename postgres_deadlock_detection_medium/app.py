#!/usr/bin/env python3

import psycopg2
import threading
import time
import sys

DB_CONFIG = {
    'host': 'localhost',
    'database': 'production_db',
    'user': 'postgres',
    'password': 'postgres'
}

def get_connection():
    """Create a new database connection"""
    return psycopg2.connect(**DB_CONFIG)

def process_order(order_id, product_id, quantity):
    """
    Process a new order by first checking/updating inventory, 
    then creating the order record.
    Transaction 1: inventory -> orders
    """
    conn = None
    try:
        conn = get_connection()
        conn.autocommit = False
        cursor = conn.cursor()
        
        print(f"[process_order] Starting order {order_id} for product {product_id}")
        
        # First access: Lock and update inventory table
        cursor.execute("""
            SELECT stock_quantity FROM inventory 
            WHERE product_id = %s 
            FOR UPDATE
        """, (product_id,))
        
        result = cursor.fetchone()
        if result and result[0] >= quantity:
            cursor.execute("""
                UPDATE inventory 
                SET stock_quantity = stock_quantity - %s 
                WHERE product_id = %s
            """, (quantity, product_id))
            
            print(f"[process_order] Updated inventory for product {product_id}")
            
            # Simulate processing time
            time.sleep(0.1)
            
            # Second access: Lock and insert into orders table
            cursor.execute("""
                INSERT INTO orders (order_id, product_id, quantity, status)
                VALUES (%s, %s, %s, 'pending')
                ON CONFLICT (order_id) DO UPDATE
                SET quantity = EXCLUDED.quantity
            """, (order_id, product_id, quantity))
            
            print(f"[process_order] Created order {order_id}")
            
            conn.commit()
            print(f"[process_order] Transaction committed for order {order_id}")
        else:
            conn.rollback()
            print(f"[process_order] Insufficient inventory for order {order_id}")
            
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"[process_order] Error: {e}")
    finally:
        if conn:
            conn.close()

def cancel_order(order_id):
    """
    Cancel an existing order by first locking the order, 
    then restoring inventory.
    Transaction 2: orders -> inventory
    """
    conn = None
    try:
        conn = get_connection()
        conn.autocommit = False
        cursor = conn.cursor()
        
        print(f"[cancel_order] Starting cancellation for order {order_id}")
        
        # First access: Lock and select from orders table
        cursor.execute("""
            SELECT product_id, quantity FROM orders 
            WHERE order_id = %s 
            FOR UPDATE
        """, (order_id,))
        
        result = cursor.fetchone()
        if result:
            product_id, quantity = result
            
            print(f"[cancel_order] Locked order {order_id}")
            
            # Simulate processing time
            time.sleep(0.1)
            
            # Second access: Lock and update inventory table
            cursor.execute("""
                UPDATE inventory 
                SET stock_quantity = stock_quantity + %s 
                WHERE product_id = %s
            """, (quantity, product_id))
            
            print(f"[cancel_order] Restored inventory for product {product_id}")
            
            # Delete or mark order as cancelled
            cursor.execute("""
                UPDATE orders 
                SET status = 'cancelled' 
                WHERE order_id = %s
            """, (order_id,))
            
            conn.commit()
            print(f"[cancel_order] Transaction committed for order {order_id}")
        else:
            conn.rollback()
            print(f"[cancel_order] Order {order_id} not found")
            
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"[cancel_order] Error: {e}")
    finally:
        if conn:
            conn.close()

def simulate_deadlock():
    """
    Simulate concurrent operations that can cause a deadlock
    """
    print("Starting deadlock simulation...")
    
    # Thread 1: Process a new order (inventory -> orders)
    t1 = threading.Thread(target=process_order, args=(101, 1, 5))
    
    # Thread 2: Cancel an existing order (orders -> inventory)
    t2 = threading.Thread(target=cancel_order, args=(100,))
    
    # Start both threads simultaneously
    t1.start()
    t2.start()
    
    # Wait for both to complete
    t1.join()
    t2.join()
    
    print("Simulation complete")

if __name__ == "__main__":
    simulate_deadlock()