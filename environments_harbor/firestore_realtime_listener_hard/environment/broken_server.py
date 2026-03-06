#!/usr/bin/env python3

import sqlite3
import json
import time
import threading

DB_PATH = '/tmp/game_state.db'

def init_database():
    """Initialize the game state database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS players')
    cursor.execute('''
        CREATE TABLE players (
            player_id TEXT PRIMARY KEY,
            x INTEGER,
            y INTEGER,
            score INTEGER,
            last_update INTEGER
        )
    ''')
    conn.commit()
    conn.close()

class StateListener:
    """Listener class for receiving state updates"""
    active_listeners = []  # Class-level list to track all listeners
    
    def __init__(self, listener_id):
        self.listener_id = listener_id
        self.updates_received = []
        StateListener.active_listeners.append(self)
    
    def on_update(self, player_id, changes):
        """Called when a player's state changes"""
        self.updates_received.append({
            'player_id': player_id,
            'changes': changes,
            'timestamp': time.time()
        })

class StateManager:
    """Manages game state and synchronization"""
    
    def __init__(self):
        self.previous_state = {}
    
    def update_player(self, player_id, x=None, y=None, score=None):
        """Update player data in database - HAS CONCURRENCY ISSUES"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Bug: Non-atomic read-modify-write without proper transaction
        cursor.execute('SELECT x, y, score FROM players WHERE player_id = ?', (player_id,))
        existing = cursor.fetchone()
        
        # Simulate some processing time that makes race conditions more likely
        time.sleep(0.001)
        
        if existing:
            current_x, current_y, current_score = existing
            new_x = x if x is not None else current_x
            new_y = y if y is not None else current_y
            new_score = score if score is not None else current_score
            
            # Bug: Using time.time() without locking can cause issues
            last_update = int(time.time())
            
            cursor.execute('''
                UPDATE players 
                SET x = ?, y = ?, score = ?, last_update = ?
                WHERE player_id = ?
            ''', (new_x, new_y, new_score, last_update, player_id))
        else:
            last_update = int(time.time())
            cursor.execute('''
                INSERT INTO players (player_id, x, y, score, last_update)
                VALUES (?, ?, ?, ?, ?)
            ''', (player_id, x or 0, y or 0, score or 0, last_update))
        
        conn.commit()
        conn.close()
        
        # Bug: Incomplete change detection - only tracks x and y, ignores score
        self._notify_listeners(player_id, {'x': x, 'y': y})
    
    def _notify_listeners(self, player_id, changes):
        """Notify listeners about state changes - BROKEN ITERATION"""
        # Bug: Iterates over list that might be modified during iteration
        # Bug: Doesn't properly filter out None values from changes
        for i in range(len(StateListener.active_listeners)):
            # Bug: Off-by-one potential or skipping listeners
            if i % 2 == 0:  # Accidentally only notifies every other listener
                listener = StateListener.active_listeners[i]
                listener.on_update(player_id, changes)
    
    def get_player(self, player_id):
        """Retrieve player data from database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT x, y, score, last_update FROM players WHERE player_id = ?', (player_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'x': result[0],
                'y': result[1],
                'score': result[2],
                'last_update': result[3]
            }
        return None
    
    def cleanup_disconnected(self, listener_id):
        """Attempt to cleanup disconnected listeners - DOESN'T WORK"""
        # Bug: Doesn't actually remove listeners from the list
        for listener in StateListener.active_listeners:
            if listener.listener_id == listener_id:
                # Bug: Just passes without removing
                pass

def simulate_concurrent_updates():
    """Simulate multiple players updating simultaneously"""
    manager = StateManager()
    
    def player_update(player_id, x, y, score):
        manager.update_player(player_id, x=x, y=y, score=score)
    
    # Create some listeners
    listeners = [StateListener(f"listener_{i}") for i in range(3)]
    
    # Simulate concurrent updates from multiple players
    threads = []
    for i in range(5):
        t = threading.Thread(target=player_update, args=(f"player_{i}", i*10, i*20, i*100))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Additional sequential updates
    for i in range(5):
        manager.update_player(f"player_{i}", x=i*10+5, y=i*20+5, score=i*100+50)
    
    return manager, listeners

def main():
    """Main test function"""
    init_database()
    
    manager, listeners = simulate_concurrent_updates()
    
    # Count total updates
    total_updates_received = sum(len(listener.updates_received) for listener in listeners)
    
    # Expected: 3 listeners * (5 concurrent + 5 sequential) = 30
    # But due to bugs, we'll receive fewer
    expected_updates = 3 * 10
    
    # Check for concurrent conflicts by looking at database state consistency
    conflicts = 0
    for i in range(5):
        player_data = manager.get_player(f"player_{i}")
        # Bug detection: if data is inconsistent or missing
        if not player_data or player_data['x'] != i*10+5:
            conflicts += 1
    
    # Try to cleanup (but it won't work)
    manager.cleanup_disconnected("listener_0")
    cleanup_successful = len(StateListener.active_listeners) == 2  # Should be 2 if cleanup worked
    
    # Output results
    results = {
        "updates_received": total_updates_received,
        "updates_expected": expected_updates,
        "concurrent_conflicts": conflicts,
        "cleanup_successful": cleanup_successful
    }
    
    with open('/home/agent/sync_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test completed. Results written to /home/agent/sync_test.json")
    print(f"Updates received: {total_updates_received}/{expected_updates}")
    print(f"Conflicts: {conflicts}")
    print(f"Cleanup successful: {cleanup_successful}")

if __name__ == '__main__':
    main()