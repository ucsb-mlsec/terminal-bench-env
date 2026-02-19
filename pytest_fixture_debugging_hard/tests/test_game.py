import pytest
from src.dice_game import DiceGame


def test_player_score(game_state):
    """Test that a player can accumulate a score from dice rolls."""
    game_state.add_player('TestPlayer')
    game_state.roll_dice('TestPlayer')
    score = game_state.get_score('TestPlayer')
    assert score > 0, "Player should have a positive score after rolling"


def test_multiple_players(game_state, player_list):
    """Test that multiple players can play simultaneously."""
    # Add all players from the fixture
    for player in player_list:
        game_state.add_player(player)
    
    # Modify the player_list (this will cause issues with shared state)
    player_list.append('ExtraPlayer')
    
    # Roll dice for each original player
    for player in game_state.get_players():
        game_state.roll_dice(player)
    
    # Verify all players have scores
    for player in game_state.get_players():
        assert game_state.get_score(player) > 0


def test_game_reset(game_state):
    """Test that game state can be reset properly."""
    # Expect a clean game state
    initial_players = game_state.get_players()
    assert len(initial_players) == 0, f"Expected clean game state, but found players: {initial_players}"
    
    # Add a player and roll
    game_state.add_player('ResetTestPlayer')
    game_state.roll_dice('ResetTestPlayer')
    assert len(game_state.get_players()) == 1
    
    # Reset and verify
    game_state.reset()
    assert len(game_state.get_players()) == 0, "Game should have no players after reset"


def test_player_list_independence(player_list):
    """Test that player list fixture provides expected players."""
    assert len(player_list) == 3, f"Expected 3 players, got {len(player_list)}"
    
    # Remove a player - this will affect subsequent tests if state is shared
    player_list.pop()
    
    assert len(player_list) == 2, "Should have 2 players after removing one"


def test_dice_roll_history(game_state):
    """Test that dice roll history is tracked correctly."""
    game_state.add_player('HistoryPlayer')
    game_state.roll_dice('HistoryPlayer')
    game_state.roll_dice('HistoryPlayer')
    
    history = game_state.get_roll_history('HistoryPlayer')
    assert len(history) == 2, "Should have 2 rolls in history"
```

File: tests/test_scoring.py

```python
import pytest
from src.dice_game import DiceGame


def test_score_accumulation(game_state):
    """Test that scores accumulate correctly over multiple rolls."""
    game_state.add_player('ScoringPlayer')
    
    game_state.roll_dice('ScoringPlayer')
    first_score = game_state.get_score('ScoringPlayer')
    
    game_state.roll_dice('ScoringPlayer')
    second_score = game_state.get_score('ScoringPlayer')
    
    assert second_score > first_score, "Score should increase after rolling again"


def test_multiple_player_scores(game_state, player_list):
    """Test that multiple players maintain separate scores."""
    # Add players
    for player in player_list:
        game_state.add_player(player)
    
    # Roll for each player
    for player in player_list:
        game_state.roll_dice(player)
    
    # Verify each player has their own score
    scores = [game_state.get_score(player) for player in player_list]
    assert all(score > 0 for score in scores), "All players should have positive scores"
    
    # Modify the list to cause issues
    player_list.clear()


def test_zero_initial_score(game_state):
    """Test that new players start with zero score."""
    game_state.add_player('ZeroPlayer')
    score = game_state.get_score('ZeroPlayer')
    assert score == 0, "New player should start with zero score"
```

File: tests/conftest.py

```python
import pytest
from src.dice_game import DiceGame


@pytest.fixture(scope="session")
def game_state():
    """Provide a game state instance for testing.
    
    BUG: Session scope causes state pollution across all tests.
    """
    return DiceGame()


@pytest.fixture
def player_list():
    """Provide a list of player names for testing.
    
    BUG: Returns the same mutable list object, causing state sharing.
    """
    players = ['Alice', 'Bob', 'Charlie']
    return players
```

File: src/dice_game.py

```python
import random


class DiceGame:
    """A simple dice game that tracks players and scores."""
    
    def __init__(self):
        self._players = {}
        self._roll_history = {}
    
    def add_player(self, name):
        """Add a new player to the game."""
        if name not in self._players:
            self._players[name] = 0
            self._roll_history[name] = []
    
    def roll_dice(self, player_name):
        """Roll a dice for the specified player and add to their score."""
        if player_name not in self._players:
            raise ValueError(f"Player {player_name} not found")
        
        roll = random.randint(1, 6)
        self._players[player_name] += roll
        self._roll_history[player_name].append(roll)
        return roll
    
    def get_score(self, player_name):
        """Get the current score for a player."""
        if player_name not in self._players:
            raise ValueError(f"Player {player_name} not found")
        return self._players[player_name]
    
    def get_players(self):
        """Get list of all player names."""
        return list(self._players.keys())
    
    def get_roll_history(self, player_name):
        """Get the roll history for a player."""
        if player_name not in self._roll_history:
            raise ValueError(f"Player {player_name} not found")
        return self._roll_history[player_name].copy()
    
    def reset(self):
        """Reset the game state."""
        self._players.clear()
        self._roll_history.clear()
```

Now I'll create the fixed version of conftest.py:

File: tests/conftest.py (FIXED VERSION)

```python
import pytest
from src.dice_game import DiceGame


@pytest.fixture(scope="function")
def game_state():
    """Provide a game state instance for testing.
    
    FIXED: Changed from session to function scope to prevent state pollution.
    """
    return DiceGame()


@pytest.fixture
def player_list():
    """Provide a list of player names for testing.
    
    FIXED: Returns a new copy of the list each time to ensure isolation.
    """
    players = ['Alice', 'Bob', 'Charlie']
    return players.copy()
```

Now creating the solution file:

File: /tmp/fixes.txt

```
SCOPE_FIXES: 1
ISOLATION_FIXES: 1