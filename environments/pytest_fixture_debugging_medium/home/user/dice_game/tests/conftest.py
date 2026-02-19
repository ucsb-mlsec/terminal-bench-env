"""
Pytest configuration and fixtures for dice game test suite.
"""
import pytest
import sqlite3
import tempfile
from pathlib import Path


@pytest.fixture(scope='session')
def db_path():
    """
    Creates a temporary database file path for testing.
    
    This fixture creates a path to a temporary database file that will be used
    throughout the test session. The database file is created in the system's
    temp directory.
    
    Scope: session - the same database path is used for all tests in the session.
    """
    temp_dir = tempfile.gettempdir()
    db_file = Path(temp_dir) / 'dice_game_test.db'
    
    yield str(db_file)
    
    # Missing cleanup code here - the database file is never removed!
    # This causes the file to persist between test runs, leading to:
    # - Stale data from previous test runs
    # - Database connection conflicts
    # - Intermittent test failures


@pytest.fixture
def db_connection(db_path):
    """
    Creates a SQLite database connection for testing.
    
    This fixture establishes a connection to the test database at the path
    provided by db_path fixture. It initializes the database schema with
    a 'games' table for tracking dice game sessions.
    
    Args:
        db_path: Path to the database file (from db_path fixture)
    
    Yields:
        sqlite3.Connection: Active database connection
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Initialize database schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            dice_roll INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    
    yield conn
    
    # Teardown: close the connection
    conn.close()


@pytest.fixture
def game_session(db_connection):
    """
    Creates a test game session in the database.
    
    This fixture sets up a game session for testing purposes. It creates
    a new session entry in the database and provides session data to tests.
    
    Args:
        db_connection: Active database connection (from db_connection fixture)
    
    Yields:
        dict: Session data including session_id and connection
    """
    import uuid
    
    session_id = str(uuid.uuid4())
    cursor = db_connection.cursor()
    
    # Insert a sample game record for the session
    cursor.execute(
        'INSERT INTO games (session_id, dice_roll) VALUES (?, ?)',
        (session_id, 6)
    )
    db_connection.commit()
    
    session_data = {
        'session_id': session_id,
        'connection': db_connection
    }
    
    yield session_data
    
    # Cleanup: remove test data for this session
    cursor.execute('DELETE FROM games WHERE session_id = ?', (session_id,))
    db_connection.commit()