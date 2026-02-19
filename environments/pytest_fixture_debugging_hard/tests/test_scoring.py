import pytest
from src.dice_game import DiceGame


def test_initial_score(game_state, initial_scores):
    """Test that players start with score 0."""
    # Add players from initial_scores dict to game_state
    for player in initial_scores.keys():
        game_state.add_player(player)
    
    # Modify initial_scores dict (causes pollution)
    initial_scores['Alice'] = 10
    
    # Assert each player starts with score 0
    for player in game_state.get_players():
        assert game_state.get_score(player) == 0, f"Player {player} should start with score 0"


def test_score_accumulation(game_state):
    """Test that scores accumulate correctly over multiple rolls."""
    # Expects clean game_state but session scope causes pollution
    player_name = 'Scorer'
    game_state.add_player(player_name)
    
    # Roll dice 3 times
    previous_score = 0
    for i in range(3):
        game_state.roll_dice(player_name)
        current_score = game_state.get_score(player_name)
        # Assert score increases after each roll
        assert current_score >= previous_score, f"Score should increase or stay same after roll {i+1}"
        previous_score = current_score
    
    # This assertion fails when other tests have already added players
    assert len(game_state.get_players()) == 1, "Should only have 1 player in clean state"


def test_shared_state_issue(initial_scores):
    """Test demonstrating shared state pollution."""
    # Assert initial state
    assert initial_scores == {'Alice': 0, 'Bob': 0}, "Should start with Alice and Bob at 0"
    
    # Modify the dict (causes pollution for subsequent tests)
    initial_scores['Charlie'] = 5
    
    # This causes subsequent tests using initial_scores to fail


def test_multiple_players(game_state, initial_scores):
    """Test adding multiple players."""
    for player in initial_scores.keys():
        game_state.add_player(player)
    
    players = game_state.get_players()
    assert 'Alice' in players
    assert 'Bob' in players


def test_score_tracking(game_state):
    """Test that scores are tracked correctly."""
    game_state.add_player('TestPlayer')
    
    # Roll and verify score changes
    game_state.roll_dice('TestPlayer')
    score = game_state.get_score('TestPlayer')
    
    assert score >= 0, "Score should be non-negative"