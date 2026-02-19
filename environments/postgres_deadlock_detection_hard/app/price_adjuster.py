#!/usr/bin/env python3

import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
import sys

class PriceAdjuster:
    """
    Adjusts product prices based on sales demand.
    This script locks tables in a consistent order to avoid deadlocks.
    """
    
    def __init__(self, db_config):
        self.db_config = db_config
    
    def get_connection(self):
        """Establish database connection"""
        return psycopg2.connect(**self.db_config)
    
    def adjust_prices_based_on_demand(self, product_id):
        """
        Adjusts product prices based on recent order volume.
        
        LOCKING ORDER (consistent, safe):
        1. First locks PRODUCTS table
        2. Then locks ORDERS table (for reading sales data)
        
        This order is safe and doesn't create circular dependencies.
        """
        conn = None
        cursor = None
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # BEGIN transaction
            conn.autocommit = False
            
            # LOCK 1: Lock the product record first (PRODUCTS table)
            # SELECT FOR UPDATE acquires row-level lock
            cursor.execute("""
                SELECT product_id, name, price, stock_quantity
                FROM products
                WHERE product_id = %s
                FOR UPDATE
            """, (product_id,))
            
            product = cursor.fetchone()
            
            if not product:
                print(f"Product {product_id} not found")
                conn.rollback()
                return False
            
            prod_id, name, current_price, stock = product
            
            # LOCK 2: Lock order records to calculate demand (ORDERS table)
            # This is second in the locking order - consistent approach
            cursor.execute("""
                SELECT COUNT(*) as order_count, SUM(quantity) as total_quantity
                FROM orders
                WHERE product_id = %s 
                AND order_date > %s
                FOR UPDATE
            """, (product_id, datetime.now() - timedelta(days=7)))
            
            demand_data = cursor.fetchone()
            order_count = demand_data[0] if demand_data[0] else 0
            total_quantity = demand_data[1] if demand_data[1] else 0
            
            # Calculate price adjustment based on demand
            new_price = current_price
            
            if total_quantity > 100:
                # High demand - increase price by 10%
                new_price = current_price * 1.10
                print(f"High demand detected ({total_quantity} units). Increasing price.")
            elif total_quantity < 10:
                # Low demand - decrease price by 5%
                new_price = current_price * 0.95
                print(f"Low demand detected ({total_quantity} units). Decreasing price.")
            else:
                print(f"Normal demand ({total_quantity} units). No price change.")
            
            # Update the product price
            if new_price != current_price:
                cursor.execute("""
                    UPDATE products
                    SET price = %s, last_updated = %s
                    WHERE product_id = %s
                """, (new_price, datetime.now(), product_id))
                
                print(f"Updated {name} price: ${current_price:.2f} -> ${new_price:.2f}")
            
            # COMMIT transaction
            conn.commit()
            return True
            
        except psycopg2.Error as e:
            print(f"Database error in price adjustment: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

def main():
    """Main execution function"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'ecommerce',
        'user': 'app_user',
        'password': 'secure_password',
        'port': 5432
    }
    
    # Initialize price adjuster
    adjuster = PriceAdjuster(db_config)
    
    # Example: Adjust prices for products 1-5
    product_ids = [1, 2, 3, 4, 5]
    
    for pid in product_ids:
        print(f"\n{'='*50}")
        print(f"Adjusting price for product ID: {pid}")
        print(f"{'='*50}")
        success = adjuster.adjust_prices_based_on_demand(pid)
        if success:
            print(f"Successfully adjusted price for product {pid}")
        else:
            print(f"Failed to adjust price for product {pid}")

if __name__ == "__main__":
    main()