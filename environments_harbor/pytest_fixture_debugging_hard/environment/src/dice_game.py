#!/usr/bin/env python3

import random


class DiceGame:
    def __init__(self):
        """Initialize a new dice game with empty state."""
        self.player_scores = {}
        self.roll_history = []
    
    def add_player(self, name: str):
        """Add a player to the game with initial score of 0.
        
        Args:
            name: The player's name
            
        Raises:
            ValueError: If player name is empty or already exists
        """
        if not name or not isinstance(name, str):
            raise ValueError("Player name must be a non-empty string")
        if name in self.player_scores:
            raise ValueError(f"Player '{name}' already exists")
        self.player_scores[name] = 0
    
    def roll_dice(self, player: str, num_dice: int = 1):
        """Roll dice for a player and update their score.
        
        Args:
            player: The player's name
            num_dice: Number of dice to roll (default 1)
            
        Returns:
            List of dice roll results
            
        Raises:
            ValueError: If player doesn't exist or num_dice is invalid
        """
        if player not in self.player_scores:
            raise ValueError(f"Player '{player}' not found")
        if num_dice < 1:
            raise ValueError("Number of dice must be at least 1")
        
        # Roll the dice
        rolls = [random.randint(1, 6) for _ in range(num_dice)]
        
        # Update score
        total = sum(rolls)
        self.player_scores[player] += total
        
        # Record in history
        self.roll_history.append((player, rolls))
        
        return rolls
    
    def get_score(self, player: str):
        """Get the current score for a player.
        
        Args:
            player: The player's name
            
        Returns:
            The player's current score
            
        Raises:
            ValueError: If player doesn't exist
        """
        if player not in self.player_scores:
            raise ValueError(f"Player '{player}' not found")
        return self.player_scores[player]
    
    def get_players(self):
        """Get list of all player names.
        
        Returns:
            List of player names
        """
        return list(self.player_scores.keys())
    
    def reset(self):
        """Clear all game state."""
        self.player_scores = {}
        self.roll_history = []
```

File: tests/conftest.py
Requirements: A pytest configuration file with broken fixtures that need to be fixed. Must include:

1. A game_state fixture with scope="session" (THIS IS INTENTIONALLY BROKEN)
2. A player_list fixture that creates a shared list without proper isolation (THIS IS INTENTIONALLY BROKEN)
3. At least one other fixture that might be used by tests
4. The fixtures should be plausible but contain the bugs described in the task

This file represents the BROKEN state that needs to be fixed.

Generate python code.

Requirements:
- Functional and complete code
- Follow all requirements
- Include necessary imports/shebang
- DO NOT wrap output in ``` markers

Output the code directly:

```python
#!/usr/bin/env python3

import pytest
from src.dice_game import DiceGame


@pytest.fixture(scope="session")
def game_state():
    """Provide a game instance for testing (BROKEN: session scope causes state pollution)."""
    game = DiceGame()
    return game


@pytest.fixture
def player_list(players=[]):
    """Provide a list of player names (BROKEN: mutable default argument)."""
    players.append("Alice")
    players.append("Bob")
    return players


@pytest.fixture
def sample_players():
    """Provide sample player names."""
    return ["Charlie", "Diana"]
```

File: tests/test_game.py
Requirements: A pytest test file with at least 3 test functions that fail when run together due to fixture issues. Tests should:

1. Use the broken fixtures from conftest.py
2. Test DiceGame functionality
3. Pass individually but fail when run together
4. Show symptoms of state pollution

Generate python code.

Requirements:
- Functional and complete code
- Follow all requirements
- Include necessary imports/shebang
- DO NOT wrap output in ``` markers

Output the code directly:

```python
#!/usr/bin/env python3

import pytest


def test_add_player(game_state):
    """Test adding a player to the game."""
    game_state.add_player("TestPlayer1")
    assert "TestPlayer1" in game_state.get_players()


def test_player_score(game_state):
    """Test getting player score."""
    game_state.add_player("TestPlayer2")
    assert game_state.get_score("TestPlayer2") == 0


def test_player_list_length(player_list):
    """Test player list has expected players."""
    assert len(player_list) == 2
    assert "Alice" in player_list


def test_roll_dice(game_state):
    """Test rolling dice."""
    game_state.add_player("TestPlayer3")
    rolls = game_state.roll_dice("TestPlayer3", 2)
    assert len(rolls) == 2
    assert all(1 <= r <= 6 for r in rolls)
```

File: tests/test_scoring.py
Requirements: A pytest test file with at least 2 test functions that use the broken fixtures. Tests should:

1. Use the broken fixtures from conftest.py
2. Test scoring functionality
3. Fail when run with other tests due to fixture issues

Generate python code.

Requirements:
- Functional and complete code
- Follow all requirements
- Include necessary imports/shebang
- DO NOT wrap output in ``` markers

Output the code directly:

```python
#!/usr/bin/env python3

import pytest


def test_initial_score(game_state):
    """Test that new players start with zero score."""
    game_state.add_player("Scorer1")
    assert game_state.get_score("Scorer1") == 0


def test_score_accumulation(game_state):
    """Test that scores accumulate correctly."""
    game_state.add_player("Scorer2")
    initial = game_state.get_score("Scorer2")
    game_state.roll_dice("Scorer2", 1)
    after_roll = game_state.get_score("Scorer2")
    assert after_roll > initial


def test_player_list_isolation(player_list):
    """Test that player list is independent."""
    assert len(player_list) == 2
    player_list.append("Eve")
    assert len(player_list) == 3
```

Now I'll create the fixed version of conftest.py and the solution file:

File: tests/conftest.py (FIXED VERSION)

```python
#!/usr/bin/env python3

import pytest
from src.dice_game import DiceGame


@pytest.fixture(scope="function")
def game_state():
    """Provide a fresh game instance for each test."""
    game = DiceGame()
    return game


@pytest.fixture
def player_list():
    """Provide a list of player names with proper isolation."""
    return ["Alice", "Bob"]


@pytest.fixture
def sample_players():
    """Provide sample player names."""
    return ["Charlie", "Diana"]
```

File: /tmp/fixes.txt

SCOPE_FIXES: 1
ISOLATION_FIXES: 1