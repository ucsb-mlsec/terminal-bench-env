#!/usr/bin/env python3
"""
Order Processor - Handles customer order processing
Locks: orders -> products (in this order)
"""

import psycopg2
from psycopg2 import sql
from decimal import Decimal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderProcessor:
    def __init__(self, db_config):
        """Initialize with database configuration"""
        self.db_config = db_config
    
    def get_connection(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def process_order(self, order_id, product_id, quantity):
        """
        Process a customer order by:
        1. Locking the order record first
        2. Then locking the product record
        3. Checking inventory availability
        4. Updating order status and product stock
        
        LOCK ORDER: orders -> products
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Begin transaction
            conn.autocommit = False
            
            logger.info(f"Processing order {order_id} for product {product_id}")
            
            # FIRST LOCK: Lock the orders table
            # This acquires a row-level lock on the specific order
            cursor.execute(
                """
                SELECT order_id, customer_id, status, total_amount
                FROM orders
                WHERE order_id = %s
                FOR UPDATE
                """,
                (order_id,)
            )
            order_row = cursor.fetchone()
            
            if not order_row:
                logger.error(f"Order {order_id} not found")
                conn.rollback()
                return False
            
            logger.info(f"Locked order record: {order_id}")
            
            # SECOND LOCK: Lock the products table
            # This acquires a row-level lock on the product
            cursor.execute(
                """
                SELECT product_id, name, price, stock_quantity
                FROM products
                WHERE product_id = %s
                FOR UPDATE
                """,
                (product_id,)
            )
            product_row = cursor.fetchone()
            
            if not product_row:
                logger.error(f"Product {product_id} not found")
                conn.rollback()
                return False
            
            logger.info(f"Locked product record: {product_id}")
            
            product_stock = product_row[3]
            product_price = product_row[2]
            
            # Check if sufficient stock is available
            if product_stock < quantity:
                logger.warning(f"Insufficient stock for product {product_id}")
                conn.rollback()
                return False
            
            # Update product stock (decrement)
            cursor.execute(
                """
                UPDATE products
                SET stock_quantity = stock_quantity - %s,
                    last_updated = CURRENT_TIMESTAMP
                WHERE product_id = %s
                """,
                (quantity, product_id)
            )
            
            # Update order status and total amount
            total_amount = Decimal(product_price) * quantity
            cursor.execute(
                """
                UPDATE orders
                SET status = 'confirmed',
                    total_amount = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE order_id = %s
                """,
                (total_amount, order_id)
            )
            
            # Commit transaction
            conn.commit()
            logger.info(f"Order {order_id} processed successfully")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                cursor.close()
                conn.close()


if __name__ == "__main__":
    # Example usage
    db_config = {
        'host': 'localhost',
        'database': 'ecommerce',
        'user': 'app_user',
        'password': 'password'
    }
    
    processor = OrderProcessor(db_config)
    processor.process_order(order_id=1001, product_id=5001, quantity=2)