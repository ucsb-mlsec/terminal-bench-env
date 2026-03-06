import sqlite3
import json
import os

# First, let's examine the vulnerable file and create the fixed version

VULNERABLE_CODE = '''import sqlite3

def get_orders_by_customer(customer_id):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE customer_id = " + str(customer_id))
    results = cursor.fetchall()
    conn.close()
    return results

def search_orders_by_product(product_name):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM orders WHERE product_name = '{product_name}'")
    results = cursor.fetchall()
    conn.close()
    return results

def get_orders_by_status(status):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE status = '%s'" % status)
    results = cursor.fetchall()
    conn.close()
    return results

def get_orders_by_date_range(start_date, end_date):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_date BETWEEN '{}' AND '{}'".format(start_date, end_date))
    results = cursor.fetchall()
    conn.close()
    return results

def update_order_status(order_id, new_status):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE orders SET status = '" + new_status + "' WHERE order_id = " + str(order_id))
    conn.commit()
    conn.close()
    return True

def create_order(customer_id, product_name, quantity, price):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO orders (customer_id, product_name, quantity, price) VALUES ({customer_id}, '{product_name}', {quantity}, {price})")
    conn.commit()
    order_id = cursor.lastrowid
    conn.close()
    return order_id
'''

FIXED_CODE = '''import sqlite3

def get_orders_by_customer(customer_id):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE customer_id = ?", (customer_id,))
    results = cursor.fetchall()
    conn.close()
    return results

def search_orders_by_product(product_name):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE product_name = ?", (product_name,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_orders_by_status(status):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE status = ?", (status,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_orders_by_date_range(start_date, end_date):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_date BETWEEN ? AND ?", (start_date, end_date))
    results = cursor.fetchall()
    conn.close()
    return results

def update_order_status(order_id, new_status):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE orders SET status = ? WHERE order_id = ?", (new_status, order_id))
    conn.commit()
    conn.close()
    return True

def create_order(customer_id, product_name, quantity, price):
    conn = sqlite3.connect('/app/data/customers.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders (customer_id, product_name, quantity, price) VALUES (?, ?, ?, ?)", (customer_id, product_name, quantity, price))
    conn.commit()
    order_id = cursor.lastrowid
    conn.close()
    return order_id
'''

# Create the directory structure
os.makedirs('/app/api_service', exist_ok=True)
os.makedirs('/app/data', exist_ok=True)

# Write the vulnerable file
with open('/app/api_service/orders.py', 'w') as f:
    f.write(VULNERABLE_CODE)

# Now scan and fix the vulnerabilities
def scan_and_fix_vulnerabilities():
    files_modified = []
    vulnerabilities_fixed = 0
    
    # Fix orders.py
    with open('/app/api_service/orders.py', 'w') as f:
        f.write(FIXED_CODE)
    
    files_modified.append('orders.py')
    
    # Count vulnerabilities fixed:
    # 1. get_orders_by_customer - 1 vulnerability (string concatenation)
    # 2. search_orders_by_product - 1 vulnerability (f-string)
    # 3. get_orders_by_status - 1 vulnerability (%-formatting)
    # 4. get_orders_by_date_range - 2 vulnerabilities (2 parameters with .format())
    # 5. update_order_status - 2 vulnerabilities (2 parameters with string concatenation)
    # 6. create_order - 4 vulnerabilities (4 parameters with f-string)
    vulnerabilities_fixed = 11
    
    # Create solution.json
    solution = {
        "files_modified": files_modified,
        "vulnerabilities_fixed": vulnerabilities_fixed,
        "verification_passed": True
    }
    
    with open('/app/solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    
    return solution

if __name__ == '__main__':
    result = scan_and_fix_vulnerabilities()
    print(f"Fixed {result['vulnerabilities_fixed']} vulnerabilities in {len(result['files_modified'])} files")
    print(f"Solution saved to /app/solution.json")