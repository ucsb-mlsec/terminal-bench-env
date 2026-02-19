import sqlite3

def get_product_by_name(product_name):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE product_name = ?", (product_name,))
    results = cursor.fetchall()
    conn.close()
    return results

def search_products_by_category(category):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE category = ?", (category,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_products_in_stock(min_quantity):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE stock_quantity > ?", (min_quantity,))
    results = cursor.fetchall()
    conn.close()
    return results

def update_product_price(product_id, new_price):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE products SET price = ? WHERE product_id = ?", (new_price, product_id))
    conn.commit()
    conn.close()
    return cursor.rowcount

def get_products_by_price_range(min_price, max_price):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE price >= ? AND price <= ?", (min_price, max_price))
    results = cursor.fetchall()
    conn.close()
    return results