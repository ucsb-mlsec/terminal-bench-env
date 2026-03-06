import time
import hashlib

def run_simulation(path):
    """
    Simulates a game from a given path through the tree.
    
    Args:
        path: list of child indices [0-2] representing moves
    
    Returns:
        1 if the path leads to a win, 0 otherwise
    
    Example: run_simulation([0, 2, 1]) simulates selecting child 0,
    then child 2 of that node, then child 1 of that node.
    """
    # Artificial delay to simulate real game computation
    time.sleep(0.05)
    
    # Deterministic win/loss determination based on the path
    # Convert path to a string and hash it
    path_str = ''.join(map(str, path))
    hash_value = int(hashlib.md5(path_str.encode()).hexdigest(), 16)
    
    # Return 1 (win) or 0 (loss) deterministically
    # Use modulo to get approximately 40-50% win rate
    return 1 if hash_value % 100 < 45 else 0