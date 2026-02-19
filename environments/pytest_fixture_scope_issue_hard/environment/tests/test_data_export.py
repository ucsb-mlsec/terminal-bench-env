#!/usr/bin/env python3

import pytest
import sqlite3


def test_export_all_customers(db_connection, clean_database, sample_data):
    """Tests exporting all customer records."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM customers")
    results = cursor.fetchall()
    
    assert len(results) == 3, f"Expected 3 customers, got {len(results)}"
    assert results is not None
    assert len(results) > 0


def test_export_filtered_customers(db_connection, clean_database):
    """Tests exporting filtered customer data."""
    cursor = db_connection.cursor()
    
    # Insert 2 customers with different email domains
    cursor.execute("INSERT INTO customers (name, email) VALUES (?, ?)", 
                   ("Test User", "testuser@test.com"))
    cursor.execute("INSERT INTO customers (name, email) VALUES (?, ?)", 
                   ("Example User", "exampleuser@example.com"))
    db_connection.commit()
    
    # Query with WHERE clause filtering by email domain
    cursor.execute("SELECT * FROM customers WHERE email LIKE '%@test.com'")
    results = cursor.fetchall()
    
    assert len(results) == 1, f"Expected 1 customer with @test.com, got {len(results)}"
    assert results[0][2] == "testuser@test.com", f"Expected testuser@test.com, got {results[0][2]}"


def test_export_empty_database(db_connection, clean_database):
    """Tests exporting from empty database."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM customers")
    results = cursor.fetchall()
    
    assert len(results) == 0, f"Expected empty database, got {len(results)} records"