#!/usr/bin/env python3

import numpy as np
import random
from evaluator import PositionEvaluator


class MCTSNode:
    def __init__(self, board, player, parent=None, move=None, prior=0.0):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visits = 0
        self.value = 0.0
        self.children = []
    
    def is_fully_expanded(self):
        """Check if all valid moves have been expanded as children"""
        return len(self.children) > 0
    
    def best_child(self, c_param=1.41):
        """Select best child using UCB1 formula"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited nodes with prior probability
                weight = float('inf')
            else:
                # UCB1 formula with prior
                exploitation = child.value / child.visits
                exploration = c_param * np.sqrt(np.log(self.visits) / child.visits)
                prior_bonus = child.prior * np.sqrt(self.visits) / (1 + child.visits)
                weight = exploitation + exploration + prior_bonus
            choices_weights.append(weight)
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self):
        """Expand node by creating children for all valid moves"""
        # This is called from _expand_node in MCTSSearch
        pass
    
    def simulate(self):
        """Run a random simulation from this node to a terminal state"""
        current_board = self.board.copy()
        current_player = self.player
        
        max_moves = 100  # Prevent infinite loops
        moves = 0
        
        while moves < max_moves:
            # Check if terminal
            winner = self._check_winner(current_board)
            if winner is not None:
                return winner
            
            # Check if board is full
            valid_moves = self._get_valid_moves(current_board)
            if len(valid_moves) == 0:
                return 0  # Draw
            
            # Make random move
            col = random.choice(valid_moves)
            current_board = self._apply_move(current_board, col, current_player)
            current_player = 3 - current_player  # Switch player (1<->2)
            moves += 1
        
        return 0  # Draw if too many moves
    
    def backpropagate(self, result):
        """Update node statistics from simulation result"""
        self.visits += 1
        # Update value based on result from this node's perspective
        if result == self.player:
            self.value += 1.0
        elif result == 0:
            self.value += 0.5
        # Loss adds 0
        
        if self.parent:
            self.parent.backpropagate(result)
    
    def _get_valid_moves(self, board):
        """Get list of valid column indices"""
        valid = []
        for col in range(7):
            if board[0, col] == 0:
                valid.append(col)
        return valid
    
    def _apply_move(self, board, column, player):
        """Apply a move and return new board state"""
        new_board = board.copy()
        for row in range(5, -1, -1):
            if new_board[row, column] == 0:
                new_board[row, column] = player
                break
        return new_board
    
    def _check_winner(self, board):
        """Check if there's a winner. Returns player number or None"""
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if board[row, col] != 0 and \
                   board[row, col] == board[row, col+1] == board[row, col+2] == board[row, col+3]:
                    return board[row, col]
        
        # Check vertical
        for row in range(3):
            for col in range(7):
                if board[row, col] != 0 and \
                   board[row, col] == board[row+1, col] == board[row+2, col] == board[row+3, col]:
                    return board[row, col]
        
        # Check diagonal (down-right)
        for row in range(3):
            for col in range(4):
                if board[row, col] != 0 and \
                   board[row, col] == board[row+1, col+1] == board[row+2, col+2] == board[row+3, col+3]:
                    return board[row, col]
        
        # Check diagonal (down-left)
        for row in range(3):
            for col in range(3, 7):
                if board[row, col] != 0 and \
                   board[row, col] == board[row+1, col-1] == board[row+2, col-2] == board[row+3, col-3]:
                    return board[row, col]
        
        return None


class MCTSSearch:
    def __init__(self, evaluator=None, simulations=1000, seed=None):
        self.evaluator = evaluator
        self.simulations = simulations
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def search(self, board, player):
        """Run MCTS and return best column to play"""
        root = MCTSNode(board, player)
        
        for _ in range(self.simulations):
            # Selection: traverse tree to find leaf node
            node = self._select(root)
            
            # Expansion: expand leaf node if not terminal
            if not self.check_terminal(node.board):
                if not node.is_fully_expanded():
                    self._expand_node(node)
                
                # Select one of the newly expanded children
                if len(node.children) > 0:
                    node = node.children[0]  # First unvisited child
            
            # Simulation: run random playout
            result = node.simulate()
            
            # Backpropagation: update statistics
            node.backpropagate(result)
        
        # Return move with most visits
        if len(root.children) == 0:
            # Fallback: return first valid move
            valid_moves = self.get_valid_moves(board)
            return valid_moves[0] if valid_moves else 0
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def _select(self, node):
        """Select a leaf node using UCB1"""
        while node.is_fully_expanded() and len(node.children) > 0:
            if self.check_terminal(node.board):
                return node
            node = node.best_child()
        return node
    
    def _expand_node(self, node):
        """Expand node by creating children for all valid moves"""
        valid_moves = self.get_valid_moves(node.board)
        
        if len(valid_moves) == 0:
            return
        
        # TODO: Get evaluation scores and convert to priors
        # Currently using uniform priors
        # FIXME: Use evaluator.evaluate() to get scores
        # FIXME: Convert scores to prior probabilities
        priors = {col: 1.0 for col in valid_moves}
        # Need to integrate evaluator here
        
        # Normalize priors to sum to 1
        total = sum(priors.values())
        for col in priors:
            priors[col] /= total
        
        # Create child nodes
        next_player = 3 - node.player  # Switch player
        for col in valid_moves:
            new_board = self.apply_move(node.board, col, node.player)
            prior = priors[col]
            child = MCTSNode(new_board, next_player, parent=node, move=col, prior=prior)
            node.children.append(child)
    
    def get_valid_moves(self, board):
        """Returns list of column indices (0-6) that aren't full"""
        valid = []
        for col in range(7):
            if board[0, col] == 0:
                valid.append(col)
        return valid
    
    def apply_move(self, board, column, player):
        """Returns new board state after move"""
        new_board = board.copy()
        for row in range(5, -1, -1):
            if new_board[row, column] == 0:
                new_board[row, column] = player
                break
        return new_board
    
    def check_terminal(self, board):
        """Checks if game is over (win/draw)"""
        winner = self.get_winner(board)
        if winner is not None:
            return True
        
        # Check if board is full
        if len(self.get_valid_moves(board)) == 0:
            return True
        
        return False
    
    def get_winner(self, board):
        """Returns winner (1, 2, or 0 for draw)"""
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if board[row, col] != 0 and \
                   board[row, col] == board[row, col+1] == board[row, col+2] == board[row, col+3]:
                    return board[row, col]
        
        # Check vertical
        for row in range(3):
            for col in range(7):
                if board[row, col] != 0 and \
                   board[row, col] == board[row+1, col] == board[row+2, col] == board[row+3, col]:
                    return board[row, col]
        
        # Check diagonal (down-right)
        for row in range(3):
            for col in range(4):
                if board[row, col] != 0 and \
                   board[row, col] == board[row+1, col+1] == board[row+2, col+2] == board[row+3, col+3]:
                    return board[row, col]
        
        # Check diagonal (down-left)
        for row in range(3):
            for col in range(3, 7):
                if board[row, col] != 0 and \
                   board[row, col] == board[row+1, col-1] == board[row+2, col-2] == board[row+3, col-3]:
                    return board[row, col]
        
        # Check for draw
        if len(self.get_valid_moves(board)) == 0:
            return 0
        
        return None