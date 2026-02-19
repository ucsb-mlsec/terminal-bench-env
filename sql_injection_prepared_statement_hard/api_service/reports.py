#!/usr/bin/env python3

import sqlite3

DB_PATH = '/app/data/customers.db'

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_customer_order_summary(customer_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT c.name, COUNT(o.order_id) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.id = " + customer_id + " GROUP BY c.name")
    result = cursor.fetchall()
    conn.close()
    return result

def get_top_customers_by_spending(limit):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT customer_id, SUM(price * quantity) as total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT {limit}")
    result = cursor.fetchall()
    conn.close()
    return result

def get_revenue_by_category(category):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT p.category, SUM(o.price * o.quantity) FROM products p JOIN orders o ON p.product_name = o.product_name WHERE p.category = '%s' GROUP BY p.category" % category)
    result = cursor.fetchall()
    conn.close()
    return result

def get_customers_by_credit_limit(min_limit, max_limit):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE credit_limit BETWEEN {} AND {}".format(min_limit, max_limit))
    result = cursor.fetchall()
    conn.close()
    return result

def search_customers_advanced(name, city, status):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM customers WHERE 1=1"
    if name:
        query += " AND name LIKE '%" + name + "%'"
    if city:
        query += " AND city = '" + city + "'"
    if status:
        query += " AND status = '" + status + "'"
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result