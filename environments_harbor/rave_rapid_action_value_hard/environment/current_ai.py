#!/usr/bin/env python3

import random
import math
from gomoku import GomokuEngine

class TreeNode:
    def __init__(self, board_state, player, parent=None, move=None):
        self.board_state = board_state
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None
        
    def uct_value(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self, c=1.41):
        return max(self.children, key=lambda child: child.uct_value(c))
    
    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visits)


class CurrentAI:
    def __init__(self):
        self.engine = GomokuEngine()
        
    def select_move(self, board_state, player, iterations=1000, seed=None):
        """
        Select the best move using basic MCTS without any optimization.
        """
        if seed is not None:
            random.seed(seed)
        
        # Initialize engine with current board state
        self.engine.set_board_state(board_state)
        
        # Check if there are any legal moves
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None
        
        # If only one move, return it
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Create root node
        root = TreeNode(board_state, player)
        root.untried_moves = legal_moves.copy()
        
        # Run MCTS iterations
        for _ in range(iterations):
            node = root
            
            # Selection - traverse tree until we find a node with untried moves
            self.engine.set_board_state(node.board_state)
            current_player = node.player
            
            while node.untried_moves is not None and len(node.untried_moves) == 0 and len(node.children) > 0:
                node = node.best_child()
                self.engine.set_board_state(node.board_state)
                current_player = 3 - current_player
            
            # Expansion - if there are untried moves, expand
            if node.untried_moves is not None and len(node.untried_moves) > 0:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                self.engine.set_board_state(node.board_state)
                new_state = self.engine.make_move(move, current_player)
                
                child_node = TreeNode(new_state, 3 - current_player, parent=node, move=move)
                
                # Get legal moves for child (inefficient - recalculates each time)
                self.engine.set_board_state(new_state)
                child_legal_moves = self.engine.get_legal_moves()
                child_node.untried_moves = child_legal_moves.copy()
                
                node.children.append(child_node)
                node = child_node
                current_player = 3 - current_player
            
            # Simulation - play out randomly from this position
            self.engine.set_board_state(node.board_state)
            winner = self.simulate(node.board_state, current_player)
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                if winner == player:
                    node.wins += 1
                elif winner == (3 - player):
                    node.wins -= 1
                node = node.parent
        
        # Return the most visited child's move
        if len(root.children) == 0:
            return random.choice(legal_moves)
        
        best_child = root.most_visited_child()
        return best_child.move
    
    def simulate(self, board_state, starting_player):
        """
        Simulate a random game from the current position.
        This is inefficient as it doesn't cache any information.
        """
        self.engine.set_board_state(board_state)
        current_player = starting_player
        
        # Check if game is already over
        winner = self.engine.check_winner()
        if winner is not None:
            return winner
        
        # Play random moves until game ends (with max depth limit)
        max_moves = 100
        moves_played = 0
        
        while moves_played < max_moves:
            legal_moves = self.engine.get_legal_moves()
            
            if not legal_moves:
                return None  # Draw
            
            # Randomly select a move (inefficient - no heuristics)
            move = random.choice(legal_moves)
            
            # Make the move
            new_state = self.engine.make_move(move, current_player)
            self.engine.set_board_state(new_state)
            
            # Check for winner
            winner = self.engine.check_winner()
            if winner is not None:
                return winner
            
            # Switch players
            current_player = 3 - current_player
            moves_played += 1
        
        # If we hit max depth, return None (draw)
        return None
    
    def evaluate_position(self, board_state, player):
        """
        Simple position evaluation (not used in basic MCTS but included).
        """
        self.engine.set_board_state(board_state)
        winner = self.engine.check_winner()
        
        if winner == player:
            return 1.0
        elif winner == (3 - player):
            return -1.0
        else:
            return 0.0


# For backward compatibility
class GomokuAI(CurrentAI):
    pass


if __name__ == "__main__":
    # Simple test
    ai = CurrentAI()
    
    # Empty board test
    empty_board = '.' * 225
    move = ai.select_move(empty_board, 1, iterations=100, seed=42)
    print(f"Selected move on empty board: {move}")
    
    # Test with some stones
    test_board = '.' * 225
    board_list = list(test_board)
    board_list[112] = 'X'  # Center
    board_list[113] = 'O'
    test_board = ''.join(board_list)
    
    move = ai.select_move(test_board, 1, iterations=100, seed=42)
    print(f"Selected move with stones: {move}")