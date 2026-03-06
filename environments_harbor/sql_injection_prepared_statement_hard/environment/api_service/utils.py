#!/usr/bin/env python3

import sqlite3

DB_PATH = '/app/data/customers.db'

def get_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)

def execute_custom_query(table_name, column_name, value):
    """Execute a custom query on a table with a single condition"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM " + table_name + " WHERE " + column_name + " = '" + value + "'")
    results = cursor.fetchall()
    conn.close()
    return results

def build_and_execute_filter(table, filters_dict):
    """Build and execute a query with multiple filter conditions"""
    conn = get_connection()
    cursor = conn.cursor()
    
    where_clauses = []
    for column, value in filters_dict.items():
        where_clauses.append(column + " = '" + str(value) + "'")
    
    where_string = " AND ".join(where_clauses)
    query = "SELECT * FROM " + table + " WHERE " + where_string
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

def get_record_count(table_name, condition):
    """Get count of records matching a condition"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {condition}")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def execute_raw_sql(sql_string):
    """Execute raw SQL string - use with caution"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql_string)
    try:
        results = cursor.fetchall()
    except:
        results = []
    conn.commit()
    conn.close()
    return results

def batch_update(table, id_list, column, value):
    """Update multiple records by ID list"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE " + table + " SET " + column + " = '" + value + "' WHERE id IN (" + id_list + ")")
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected