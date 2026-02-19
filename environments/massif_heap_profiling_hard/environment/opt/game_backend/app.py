#!/usr/bin/env python3
"""
Multiplayer Resource Management Game Backend
A Flask-based web application for managing game sessions, players, and resources.
"""

from flask import Flask, request, jsonify
from datetime import datetime
import uuid
import random

app = Flask(__name__)

# Global storage for game sessions
game_sessions = {}


class TradeHistory:
    """
    Logs all trading activities between players.
    Maintains a complete history for analytics and dispute resolution.
    """
    
    def __init__(self):
        self.trades = []
        self.trade_count = 0
    
    def log_trade(self, from_player, to_player, resource, amount, timestamp):
        """Log a trade transaction with full details."""
        trade_record = {
            'id': self.trade_count,
            'from': from_player,
            'to': to_player,
            'resource': resource,
            'amount': amount,
            'timestamp': timestamp,
            'metadata': {
                'session_state': None  # Will be populated
            }
        }
        self.trades.append(trade_record)
        self.trade_count += 1
        return trade_record


class PlayerInventory:
    """
    Manages individual player inventory with detailed item tracking.
    Supports various resource types and maintains historical data.
    """
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.items = {}
        self.transaction_history = []
        self.snapshots = []  # Historical inventory snapshots
    
    def add_item(self, resource_type, amount):
        """
        Add items to inventory with full transaction logging.
        Maintains detailed history for each addition.
        """
        # Create detailed transaction record
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'resource': resource_type,
            'amount': amount,
            'action': 'add',
            'inventory_snapshot': dict(self.items)  # Store current state
        }
        self.transaction_history.append(transaction)
        
        # Update inventory
        if resource_type not in self.items:
            self.items[resource_type] = 0
        self.items[resource_type] += amount
        
        # Store snapshot every time for audit purposes
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'items': dict(self.items),
            'transaction_ref': len(self.transaction_history) - 1
        }
        self.snapshots.append(snapshot)
        
        return self.items[resource_type]
    
    def remove_item(self, resource_type, amount):
        """Remove items from inventory with transaction logging."""
        if resource_type in self.items and self.items[resource_type] >= amount:
            transaction = {
                'timestamp': datetime.now().isoformat(),
                'resource': resource_type,
                'amount': amount,
                'action': 'remove',
                'inventory_snapshot': dict(self.items)
            }
            self.transaction_history.append(transaction)
            
            self.items[resource_type] -= amount
            
            # Store snapshot
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'items': dict(self.items),
                'transaction_ref': len(self.transaction_history) - 1
            }
            self.snapshots.append(snapshot)
            
            return True
        return False
    
    def get_items(self):
        """Return current inventory state."""
        return self.items


class ResourceManager:
    """
    Handles resource generation and distribution across the game world.
    Manages resource pools and regeneration rates.
    """
    
    def __init__(self):
        self.resource_types = ['wood', 'stone', 'iron', 'gold', 'gems']
        self.generation_rates = {
            'wood': 5,
            'stone': 4,
            'iron': 3,
            'gold': 2,
            'gems': 1
        }
        self.resource_pools = {rt: 1000 for rt in self.resource_types}
    
    def generate_resource(self):
        """Generate a random resource based on availability and rates."""
        available = [r for r in self.resource_types if self.resource_pools[r] > 0]
        if not available:
            return None, 0
        
        resource = random.choice(available)
        amount = random.randint(1, self.generation_rates[resource])
        
        if self.resource_pools[resource] >= amount:
            self.resource_pools[resource] -= amount
            return resource, amount
        
        return None, 0
    
    def replenish_pools(self):
        """Replenish resource pools over time."""
        for resource in self.resource_types:
            self.resource_pools[resource] = min(
                1000, 
                self.resource_pools[resource] + self.generation_rates[resource]
            )


class GameSession:
    """
    Manages a complete game session including players, resources, and game state.
    Coordinates all game activities and maintains session integrity.
    """
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.players = {}
        self.resource_manager = ResourceManager()
        self.trade_history = TradeHistory()
        self.created_at = datetime.now()
        self.event_log = []
        self.state_history = []
        self.active = True
    
    def add_player(self, player_name):
        """Add a new player to the game session."""
        player_id = str(uuid.uuid4())
        inventory = PlayerInventory(player_id)
        
        self.players[player_id] = {
            'name': player_name,
            'inventory': inventory,
            'joined_at': datetime.now().isoformat(),
            'stats': {
                'resources_collected': 0,
                'trades_completed': 0
            }
        }
        
        # Log event
        event = {
            'type': 'player_joined',
            'player_id': player_id,
            'player_name': player_name,
            'timestamp': datetime.now().isoformat(),
            'session_state': self._capture_state()
        }
        self.event_log.append(event)
        
        return player_id
    
    def collect_resource(self, player_id):
        """Allow a player to collect resources from the game world."""
        if player_id not in self.players:
            return None, "Player not found"
        
        resource, amount = self.resource_manager.generate_resource()
        
        if resource is None:
            return None, "No resources available"
        
        player = self.players[player_id]
        player['inventory'].add_item(resource, amount)
        player['stats']['resources_collected'] += amount
        
        # Log detailed event
        event = {
            'type': 'resource_collected',
            'player_id': player_id,
            'resource': resource,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'session_state': self._capture_state()
        }
        self.event_log.append(event)
        
        return resource, amount
    
    def execute_trade(self, from_player_id, to_player_id, resource, amount):
        """Execute a trade between two players."""
        if from_player_id not in self.players or to_player_id not in self.players:
            return False, "One or both players not found"
        
        from_player = self.players[from_player_id]
        to_player = self.players[to_player_id]
        
        # Check if from_player has the resource
        if not from_player['inventory'].remove_item(resource, amount):
            return False, "Insufficient resources"
        
        # Add to recipient
        to_player['inventory'].add_item(resource, amount)
        
        # Update stats
        from_player['stats']['trades_completed'] += 1
        to_player['stats']['trades_completed'] += 1
        
        # Log trade
        trade_record = self.trade_history.log_trade(
            from_player_id,
            to_player_id,
            resource,
            amount,
            datetime.now().isoformat()
        )
        
        # Store session state in trade record
        trade_record['metadata']['session_state'] = self._capture_state()
        
        # Log event
        event = {
            'type': 'trade_executed',
            'from_player': from_player_id,
            'to_player': to_player_id,
            'resource': resource,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'session_state': self._capture_state()
        }
        self.event_log.append(event)
        
        return True, "Trade successful"
    
    def _capture_state(self):
        """Capture complete current game state for historical records."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'players': {},
            'resource_pools': dict(self.resource_manager.resource_pools),
            'event_count': len(self.event_log),
            'trade_count': self.trade_history.trade_count
        }
        
        for pid, player in self.players.items():
            state['players'][pid] = {
                'name': player['name'],
                'inventory': dict(player['inventory'].get_items()),
                'stats': dict(player['stats'])
            }
        
        # Store in history
        self.state_history.append(state)
        
        return state
    
    def get_session_info(self):
        """Return current session information."""
        return {
            'session_id': self.session_id,
            'player_count': len(self.players),
            'total_events': len(self.event_log),
            'total_trades': self.trade_history.trade_count,
            'active': self.active
        }


@app.route('/start_game', methods=['POST'])
def start_game():
    """Initialize a new game session."""
    session_id = str(uuid.uuid4())
    game_sessions[session_id] = GameSession(session_id)
    
    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'message': 'Game session created'
    }), 201


@app.route('/add_player', methods=['POST'])
def add_player():
    """Add a player to an existing game session."""
    data = request.get_json()
    session_id = data.get('session_id')
    player_name = data.get('player_name')
    
    if not session_id or not player_name:
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400
    
    if session_id not in game_sessions:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    player_id = session.add_player(player_name)
    
    return jsonify({
        'status': 'success',
        'player_id': player_id,
        'player_name': player_name
    }), 201


@app.route('/collect_resource', methods=['POST'])
def collect_resource():
    """Handle resource collection by a player."""
    data = request.get_json()
    session_id = data.get('session_id')
    player_id = data.get('player_id')
    
    if not session_id or not player_id:
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400
    
    if session_id not in game_sessions:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    resource, result = session.collect_resource(player_id)
    
    if resource is None:
        return jsonify({'status': 'error', 'message': result}), 400
    
    return jsonify({
        'status': 'success',
        'resource': resource,
        'amount': result
    }), 200


@app.route('/trade', methods=['POST'])
def trade():
    """Handle trading between players."""
    data = request.get_json()
    session_id = data.get('session_id')
    from_player = data.get('from_player')
    to_player = data.get('to_player')
    resource = data.get('resource')
    amount = data.get('amount')
    
    if not all([session_id, from_player, to_player, resource, amount]):
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400
    
    if session_id not in game_sessions:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    success, message = session.execute_trade(from_player, to_player, resource, amount)
    
    if not success:
        return jsonify({'status': 'error', 'message': message}), 400
    
    return jsonify({
        'status': 'success',
        'message': message
    }), 200


@app.route('/session_info', methods=['GET'])
def session_info():
    """Get information about a game session."""
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Missing session_id'}), 400
    
    if session_id not in game_sessions:
        return jsonify({'status': 'error', 'message': 'Session not found'}), 404
    
    session = game_sessions[session_id]
    info = session.get_session_info()
    
    return jsonify({
        'status': 'success',
        'session': info
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)