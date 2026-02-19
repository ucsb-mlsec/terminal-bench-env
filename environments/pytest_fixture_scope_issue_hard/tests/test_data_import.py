import pytest
import sqlite3


def test_import_single_customer(db_connection, clean_database):
    """Tests inserting one customer record into the database."""
    cursor = db_connection.cursor()
    
    # Insert a single customer
    cursor.execute(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        (10, 'John Doe', 'john@example.com')
    )
    db_connection.commit()
    
    # Verify the record was inserted
    cursor.execute("SELECT * FROM customers")
    results = cursor.fetchall()
    
    assert len(results) == 1


def test_import_multiple_customers(db_connection, clean_database):
    """Tests inserting multiple customer records."""
    cursor = db_connection.cursor()
    
    # Insert 3 customers
    customers = [
        (20, 'Alice Smith', 'alice@example.com'),
        (21, 'Bob Johnson', 'bob@example.com'),
        (22, 'Charlie Brown', 'charlie@example.com')
    ]
    
    for customer in customers:
        cursor.execute(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            customer
        )
    db_connection.commit()
    
    # Verify all records
    cursor.execute("SELECT * FROM customers")
    results = cursor.fetchall()
    
    assert len(results) == 3


def test_import_duplicate_handling(db_connection, clean_database):
    """Tests handling duplicate customer IDs."""
    cursor = db_connection.cursor()
    
    # Insert first customer
    cursor.execute(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        (30, 'David Lee', 'david@example.com')
    )
    db_connection.commit()
    
    # Attempt to insert duplicate
    try:
        cursor.execute(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            (30, 'David Duplicate', 'duplicate@example.com')
        )
        db_connection.commit()
    except sqlite3.IntegrityError:
        pass  # Expected behavior
    
    # Verify only 1 record exists
    cursor.execute("SELECT * FROM customers WHERE id = 30")
    results = cursor.fetchall()
    
    assert len(results) == 1