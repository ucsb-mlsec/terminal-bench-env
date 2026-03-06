import pytest
import sqlite3
import os


@pytest.fixture(scope='session')
def db_path():
    """Fixture providing database path - PROBLEM: scope should be 'function'"""
    return '/tmp/test_db.sqlite'


@pytest.fixture(scope='session')
def db_connection(db_path):
    """Fixture providing database connection - PROBLEMS: scope should be 'function', missing cleanup"""
    connection = sqlite3.connect(db_path)
    return connection


@pytest.fixture(scope='module')
def db_cursor(db_connection):
    """Fixture providing database cursor - PROBLEMS: scope should be 'function', missing cleanup"""
    cursor = db_connection.cursor()
    return cursor


@pytest.fixture(scope='function')
def clean_database(db_connection):
    """Fixture for clean database with customers table - PROBLEM: missing cleanup/teardown"""
    db_connection.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    ''')
    db_connection.commit()
    yield db_connection


@pytest.fixture(scope='module')
def sample_data(db_connection):
    """Fixture providing sample data - PROBLEMS: scope should be 'function', missing cleanup"""
    db_connection.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    ''')
    
    customers = [
        (1, 'John Doe', 'john@example.com'),
        (2, 'Jane Smith', 'jane@example.com'),
        (3, 'Bob Johnson', 'bob@example.com')
    ]
    
    db_connection.executemany(
        'INSERT OR REPLACE INTO customers (id, name, email) VALUES (?, ?, ?)',
        customers
    )
    db_connection.commit()
    return db_connection