#!/usr/bin/env python3
import pytest
import sqlite3


def test_database_connection(db_connection):
    """Test that the database connection is valid."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result == (1,)


def test_games_table_exists(db_connection):
    """Test that the games table exists with proper schema."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == 'games'


def test_insert_game_data(db_connection):
    """Test inserting and retrieving game data."""
    cursor = db_connection.cursor()
    
    # Insert test game data
    cursor.execute("INSERT INTO games (player_name, score, dice_roll) VALUES (?, ?, ?)", 
                   ('TestPlayer', 100, 6))
    db_connection.commit()
    
    # Retrieve and verify
    cursor.execute("SELECT player_name, score, dice_roll FROM games WHERE player_name = ?", 
                   ('TestPlayer',))
    result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == 'TestPlayer'
    assert result[1] == 100
    assert result[2] == 6