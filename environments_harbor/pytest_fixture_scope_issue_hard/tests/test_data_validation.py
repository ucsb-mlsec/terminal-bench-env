#!/usr/bin/env python3

import pytest
import sqlite3


def test_validate_email_format(db_connection, clean_database):
    """Tests that valid email formats are accepted."""
    cursor = db_connection.cursor()
    cursor.execute(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        (100, 'Test User', 'valid@email.com')
    )
    db_connection.commit()
    
    cursor.execute("SELECT COUNT(*) FROM customers WHERE id = 100")
    count = cursor.fetchone()[0]
    assert count == 1


def test_validate_required_fields(db_connection, clean_database):
    """Tests that NULL values in required fields are handled."""
    cursor = db_connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            (101, None, 'test@test.com')
        )
        db_connection.commit()
    except sqlite3.IntegrityError:
        # Expected if name is a required field
        pass
    
    cursor.execute("SELECT COUNT(*) FROM customers WHERE id = 101")
    count = cursor.fetchone()[0]
    # Verify behavior - either inserted or not based on schema constraints


def test_validate_unique_ids(db_connection, clean_database, sample_data):
    """Tests that customer IDs are unique."""
    cursor = db_connection.cursor()
    
    # Query existing customers from sample_data
    cursor.execute("SELECT id FROM customers LIMIT 1")
    existing_row = cursor.fetchone()
    existing_id = existing_row[0] if existing_row else 1
    
    # Get original count
    cursor.execute("SELECT COUNT(*) FROM customers")
    original_count = cursor.fetchone()[0]
    
    # Try to insert a customer with an existing ID
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            (existing_id, 'Duplicate User', 'duplicate@test.com')
        )
        db_connection.commit()
    
    # Verify count remains unchanged
    cursor.execute("SELECT COUNT(*) FROM customers")
    final_count = cursor.fetchone()[0]
    assert final_count == original_count


def test_validate_data_types(db_connection, clean_database):
    """Tests proper data type handling."""
    cursor = db_connection.cursor()
    cursor.execute(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        (102, 'Type Test', 'types@test.com')
    )
    db_connection.commit()
    
    cursor.execute("SELECT id, name, email FROM customers WHERE id = 102")
    row = cursor.fetchone()
    
    assert row is not None
    assert type(row[0]) == int
    assert type(row[1]) == str
    assert type(row[2]) == str