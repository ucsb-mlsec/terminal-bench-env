#!/usr/bin/env python3

import psycopg2
from psycopg2 import sql
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'ecommerce_db',
    'user': 'app_user',
    'password': 'app_password'
}

def get_db_connection():
    """Establish database connection"""
    return psycopg2.connect(**DB_CONFIG)

def update_inventory_after_shipment(product_id, quantity_shipped):
    """
    Update inventory levels after products are shipped.
    
    This function:
    1. First locks the products table to get product details
    2. Then locks the inventory table to update stock levels
    
    This locking order (products -> inventory) may conflict with other scripts
    that lock these tables in different order.
    """
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Begin transaction
        conn.autocommit = False
        logger.info(f"Starting inventory update for product {product_id}")
        
        # FIRST LOCK: Lock products table to get product information
        # Using SELECT FOR UPDATE to acquire exclusive row lock
        cursor.execute("""
            SELECT product_id, name, category, reorder_threshold
            FROM products
            WHERE product_id = %s
            FOR UPDATE
        """, (product_id,))
        
        product = cursor.fetchone()
        if not product:
            raise ValueError(f"Product {product_id} not found")
        
        logger.info(f"Locked product record: {product[1]}")
        
        # SECOND LOCK: Lock inventory table to update stock levels
        # This creates the potential for deadlock if another script
        # locks inventory first, then products
        cursor.execute("""
            SELECT inventory_id, product_id, quantity_available, warehouse_location
            FROM inventory
            WHERE product_id = %s
            FOR UPDATE
        """, (product_id,))
        
        inventory = cursor.fetchone()
        if not inventory:
            raise ValueError(f"Inventory record not found for product {product_id}")
        
        logger.info(f"Locked inventory record for product {product_id}")
        
        # Update inventory quantity
        new_quantity = inventory[2] - quantity_shipped
        
        if new_quantity < 0:
            raise ValueError(f"Insufficient inventory. Available: {inventory[2]}, Requested: {quantity_shipped}")
        
        cursor.execute("""
            UPDATE inventory
            SET quantity_available = %s,
                last_updated = CURRENT_TIMESTAMP
            WHERE product_id = %s
        """, (new_quantity, product_id))
        
        logger.info(f"Updated inventory: {inventory[2]} -> {new_quantity}")
        
        # Check if reorder is needed
        reorder_threshold = product[3]
        if new_quantity < reorder_threshold:
            logger.warning(f"Product {product_id} below reorder threshold. Current: {new_quantity}, Threshold: {reorder_threshold}")
        
        # Commit transaction
        conn.commit()
        logger.info(f"Transaction committed successfully for product {product_id}")
        
        return True
        
    except Exception as e:
        if conn:
            conn.rollback()
            logger.error(f"Transaction rolled back due to error: {str(e)}")
        raise
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def bulk_inventory_update(product_updates):
    """
    Bulk update inventory for multiple products.
    
    Locks: products table first, then inventory table for each product
    """
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        conn.autocommit = False
        
        for product_id, quantity_change in product_updates:
            # Lock products first
            cursor.execute("""
                SELECT product_id FROM products WHERE product_id = %s FOR UPDATE
            """, (product_id,))
            
            # Then lock inventory
            cursor.execute("""
                UPDATE inventory 
                SET quantity_available = quantity_available + %s
                WHERE product_id = %s
            """, (quantity_change, product_id))
        
        conn.commit()
        logger.info(f"Bulk update completed for {len(product_updates)} products")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Bulk update failed: {str(e)}")
        raise
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    # Example usage
    try:
        update_inventory_after_shipment(product_id=101, quantity_shipped=5)
    except Exception as e:
        logger.error(f"Failed to update inventory: {str(e)}")