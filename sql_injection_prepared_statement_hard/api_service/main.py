#!/usr/bin/env python3

import sqlite3

# Flask-like route decorator comments
# @app.route('/api/customer/<customer_id>', methods=['GET'])
def get_customer_by_id(customer_id):
    """Get customer by ID - VULNERABLE to SQL injection via string concatenation"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Direct string concatenation
    query = "SELECT * FROM customers WHERE id = " + customer_id
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    return result


# @app.route('/api/customers/search', methods=['GET'])
def search_customers_by_name(name):
    """Search customers by name - VULNERABLE to SQL injection via f-string"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: f-string formatting
    query = f"SELECT * FROM customers WHERE name LIKE '%{name}%'"
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    return results


# @app.route('/api/customers/city/<city>', methods=['GET'])
def get_customers_by_city(city):
    """Get customers by city - VULNERABLE to SQL injection via %-formatting"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: %-formatting
    query = "SELECT * FROM customers WHERE city = '%s'" % city
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    return results


# @app.route('/api/customer/<customer_id>/email', methods=['PUT'])
def update_customer_email(customer_id, email):
    """Update customer email - VULNERABLE to SQL injection via .format()"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: .format() method
    query = "UPDATE customers SET email = '{}' WHERE id = {}".format(email, customer_id)
    cursor.execute(query)
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected


# @app.route('/api/customer/<customer_id>', methods=['DELETE'])
def delete_customer(customer_id):
    """Delete customer - VULNERABLE to SQL injection via string concatenation"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Direct string concatenation
    query = "DELETE FROM customers WHERE id = " + str(customer_id)
    cursor.execute(query)
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected


# @app.route('/api/customers/filter', methods=['POST'])
def get_customers_by_status_and_country(status, country):
    """Get customers by status and country - VULNERABLE to SQL injection"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Mixed string concatenation and f-string
    query = "SELECT * FROM customers WHERE status = '" + status + f"' AND country = '{country}'"
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    return results


# @app.route('/api/customers/order', methods=['GET'])
def get_customers_ordered(order_by):
    """Get customers with custom ordering - VULNERABLE to SQL injection"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Direct concatenation of ORDER BY clause
    query = "SELECT * FROM customers ORDER BY " + order_by
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    return results


# @app.route('/api/customer/login', methods=['POST'])
def authenticate_customer(username, password):
    """Authenticate customer - VULNERABLE to SQL injection"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Classic SQL injection in authentication
    query = f"SELECT * FROM customers WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    
    result = cursor.fetchone()
    conn.close()
    return result


# @app.route('/api/customer/<customer_id>/address', methods=['PUT'])
def update_customer_address(customer_id, street, city, zipcode):
    """Update customer address - VULNERABLE to SQL injection"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Multiple vulnerable parameters
    query = "UPDATE customers SET street = '%s', city = '%s', zipcode = '%s' WHERE id = %s" % (street, city, zipcode, customer_id)
    cursor.execute(query)
    
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected


# @app.route('/api/customers/range', methods=['GET'])
def get_customers_in_id_range(start_id, end_id):
    """Get customers in ID range - VULNERABLE to SQL injection"""
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    
    # VULNERABLE: Both parameters concatenated
    query = "SELECT * FROM customers WHERE id >= " + start_id + " AND id <= " + end_id
    cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    return results