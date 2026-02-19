import pytest
import sys
import os.path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dice_game import DiceGame

# BUG: session scope causes state pollution across all tests
@pytest.fixture(scope="session")
def game_state():
    """Provides a game state instance - BUGGY: session scope causes sharing"""
    return DiceGame()

# BUG: Returns shared mutable list that persists between tests
_shared_list = ['Player1', 'Player2', 'Player3']

@pytest.fixture
def player_list():
    """Provides a list of player names - BUGGY: shared mutable object"""
    return _shared_list

# BUG: Returns same dict instance without copying
_scores = {'Alice': 0, 'Bob': 0}

@pytest.fixture
def initial_scores():
    """Provides initial score dictionary - BUGGY: shared dict reference"""
    return _scores