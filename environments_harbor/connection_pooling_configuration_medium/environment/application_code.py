#!/usr/bin/env python3
"""
E-commerce Platform Web Application

This application serves an e-commerce platform with variable traffic patterns.
Peak traffic occurs during business hours (9 AM - 6 PM) and flash sales.
The application handles product browsing, inventory checks, order processing,
and analytics queries.

Traffic patterns:
- Normal load: 30-50 requests per minute
- Peak load: 100-150 requests per minute
- Flash sale events: Up to 300 requests per minute for short bursts
"""

import asyncio
import json
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import asyncpg


# Global connection pool
db_pool: Optional[asyncpg.Pool] = None


async def init_connection_pool(config_path: str = "/app/current_config.json"):
    """
    Initialize the database connection pool with configuration from JSON file.
    
    Configuration parameters:
    - min_size: Minimum number of connections to maintain
    - max_size: Maximum number of connections allowed
    - timeout_seconds: How long to wait for an available connection
    """
    global db_pool
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    db_pool = await asyncpg.create_pool(
        host='postgres-db.internal',
        port=5432,
        user='app_user',
        password='app_password',
        database='ecommerce',
        min_size=config['min_size'],
        max_size=config['max_size'],
        timeout=config['timeout_seconds']
    )
    
    return db_pool


async def close_connection_pool():
    """Properly close the database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Initialize connection pool on application startup."""
    await init_connection_pool()


@app.on_event("shutdown")
async def shutdown_event():
    """Close connection pool on application shutdown."""
    await close_connection_pool()


@app.get("/products/{product_id}")
async def get_product(product_id: int):
    """
    Retrieve product details by ID.
    
    Usage pattern: Very high traffic endpoint
    - Receives 50-100 requests per minute during peak hours
    - Query duration: ~50ms (fast, indexed query)
    - Expected concurrency: 5-15 simultaneous requests during peak
    """
    # Proper connection handling with context manager
    async with db_pool.acquire() as connection:
        product = await connection.fetchrow(
            "SELECT * FROM products WHERE id = $1",
            product_id
        )
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return dict(product)


@app.get("/products")
async def list_products(category: Optional[str] = None, limit: int = 20):
    """
    List products with optional category filter.
    
    Usage pattern: High traffic endpoint
    - Receives 30-60 requests per minute
    - Query duration: ~80ms (fast, uses indexes)
    - Expected concurrency: 3-10 simultaneous requests
    """
    async with db_pool.acquire() as connection:
        if category:
            products = await connection.fetch(
                "SELECT * FROM products WHERE category = $1 LIMIT $2",
                category, limit
            )
        else:
            products = await connection.fetch(
                "SELECT * FROM products ORDER BY created_at DESC LIMIT $1",
                limit
            )
        return [dict(p) for p in products]


@app.get("/inventory/check/{product_id}")
async def check_inventory(product_id: int):
    """
    Check real-time inventory for a product across warehouses.
    
    Usage pattern: Medium traffic, medium duration
    - Receives 20-40 requests per minute
    - Query duration: ~300ms (joins multiple tables)
    - Expected concurrency: 3-8 simultaneous requests
    """
    async with db_pool.acquire() as connection:
        # Simulating a medium-complexity query with joins
        inventory = await connection.fetch(
            """
            SELECT w.name, i.quantity, i.reserved 
            FROM inventory i 
            JOIN warehouses w ON i.warehouse_id = w.id
            WHERE i.product_id = $1
            """,
            product_id
        )
        return [dict(i) for i in inventory]


@app.post("/orders")
async def create_order(order_data: dict):
    """
    Create a new order (transactional operation).
    
    Usage pattern: Medium-high traffic, variable duration
    - Receives 15-40 requests per minute during peak
    - Query duration: ~400ms (transaction with multiple inserts/updates)
    - Expected concurrency: 3-10 simultaneous requests
    - Uses transactions, holds connection longer
    """
    async with db_pool.acquire() as connection:
        # Proper transaction handling with try/finally
        async with connection.transaction():
            try:
                # Insert order
                order = await connection.fetchrow(
                    "INSERT INTO orders (user_id, total) VALUES ($1, $2) RETURNING *",
                    order_data['user_id'], order_data['total']
                )
                
                # Insert order items
                for item in order_data['items']:
                    await connection.execute(
                        "INSERT INTO order_items (order_id, product_id, quantity) VALUES ($1, $2, $3)",
                        order['id'], item['product_id'], item['quantity']
                    )
                
                # Update inventory
                for item in order_data['items']:
                    await connection.execute(
                        "UPDATE inventory SET quantity = quantity - $1 WHERE product_id = $2",
                        item['quantity'], item['product_id']
                    )
                
                return dict(order)
            except Exception as e:
                # Transaction will auto-rollback
                raise HTTPException(status_code=400, detail=str(e))


@app.get("/analytics/sales")
async def get_sales_analytics(days: int = 30):
    """
    Generate sales analytics report.
    
    Usage pattern: Low traffic, long duration
    - Receives 2-5 requests per minute
    - Query duration: ~1500ms (complex aggregations, large dataset)
    - Expected concurrency: 1-3 simultaneous requests
    - This is a heavy query that can hold connections for extended periods
    """
    async with db_pool.acquire() as connection:
        # Complex analytical query with aggregations
        analytics = await connection.fetch(
            """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as order_count,
                SUM(total) as revenue,
                AVG(total) as avg_order_value
            FROM orders
            WHERE created_at > NOW() - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """,
            days
        )
        return [dict(a) for a in analytics]


@app.post("/batch/import")
async def batch_import_products(products: list):
    """
    Batch import products (admin operation).
    
    Usage pattern: Rare, very long duration
    - Receives 1-2 requests per hour (admin only)
    - Query duration: ~2000ms+ (bulk inserts)
    - Expected concurrency: 1 request at a time
    - Holds connection for extended period during bulk operations
    """
    async with db_pool.acquire() as connection:
        async with connection.transaction():
            try:
                results = []
                for product in products:
                    result = await connection.fetchrow(
                        """
                        INSERT INTO products (name, description, price, category)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                        """,
                        product['name'], product['description'], 
                        product['price'], product['category']
                    )
                    results.append(dict(result))
                
                return {"imported": len(results), "ids": results}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Usage pattern: Very high frequency, very fast
    - Receives 100+ requests per minute (monitoring systems)
    - Query duration: ~20ms (simple SELECT 1)
    - Expected concurrency: 2-5 simultaneous requests
    """
    try:
        async with db_pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)