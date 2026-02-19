import pytest
import sqlite3


def test_new_game_session(db_connection):
    """Test that a new game session can be created."""
    cursor = db_connection.cursor()
    cursor.execute("INSERT INTO games (player_name, score) VALUES (?, ?)", ("Alice", 100))
    db_connection.commit()
    
    cursor.execute("SELECT * FROM games WHERE player_name = ?", ("Alice",))
    result = cursor.fetchone()
    assert result is not None
    assert result[1] == "Alice"
    assert result[2] == 100


def test_dice_roll_range(game_session):
    """Test that dice rolls are within valid range."""
    cursor = game_session.cursor()
    cursor.execute("INSERT INTO dice_rolls (game_id, roll_value) VALUES (?, ?)", (1, 5))
    game_session.commit()
    
    cursor.execute("SELECT roll_value FROM dice_rolls WHERE game_id = 1")
    roll = cursor.fetchone()[0]
    assert 1 <= roll <= 6


def test_game_count(db_connection):
    """Test that game count is correct - fails if old data persists."""
    cursor = db_connection.cursor()
    
    # Insert one game
    cursor.execute("INSERT INTO games (player_name, score) VALUES (?, ?)", ("Bob", 150))
    db_connection.commit()
    
    # Count all games - should be exactly 1 on first run
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    
    # This will fail on second run if database isn't cleaned up
    assert count == 1, f"Expected 1 game, but found {count}"


def test_session_isolation(game_session):
    """Test that each game session has isolated data."""
    cursor = game_session.cursor()
    
    # Check that games table is empty for this session
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    
    # Insert a test game
    cursor.execute("INSERT INTO games (player_name, score) VALUES (?, ?)", ("Charlie", 200))
    game_session.commit()
    
    # Verify the insert
    cursor.execute("SELECT COUNT(*) FROM games")
    new_count = cursor.fetchone()[0]
    assert new_count == count + 1


def test_multiple_dice_rolls(game_session):
    """Test multiple dice rolls in a game."""
    cursor = game_session.cursor()
    
    # Insert a game first
    cursor.execute("INSERT INTO games (player_name, score) VALUES (?, ?)", ("Dave", 75))
    game_session.commit()
    
    cursor.execute("SELECT id FROM games WHERE player_name = ?", ("Dave",))
    game_id = cursor.fetchone()[0]
    
    # Add multiple dice rolls
    for roll in [3, 6, 2, 4]:
        cursor.execute("INSERT INTO dice_rolls (game_id, roll_value) VALUES (?, ?)", (game_id, roll))
    game_session.commit()
    
    # Verify all rolls
    cursor.execute("SELECT COUNT(*) FROM dice_rolls WHERE game_id = ?", (game_id,))
    roll_count = cursor.fetchone()[0]
    assert roll_count == 4