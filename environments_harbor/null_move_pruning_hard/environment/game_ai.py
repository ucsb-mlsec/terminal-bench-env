#!/usr/bin/env python3

import random
import copy

class GameState:
    """
    A two-player abstract strategy game where players place pieces on a 6x6 board
    trying to form patterns and control territory for points.
    """
    
    def __init__(self, board=None, current_player=1, move_count=0):
        self.size = 6
        if board is None:
            self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        else:
            self.board = board
        self.current_player = current_player  # 1 or 2
        self.move_count = move_count
        self.max_moves = 30
    
    def get_legal_moves(self):
        """Returns list of legal moves (row, col) tuples for empty squares."""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:
                    moves.append((row, col))
        return moves
    
    def make_move(self, move):
        """Returns a new GameState with the move applied."""
        row, col = move
        new_board = copy.deepcopy(self.board)
        new_board[row][col] = self.current_player
        next_player = 3 - self.current_player  # Switch between 1 and 2
        return GameState(new_board, next_player, self.move_count + 1)
    
    def is_game_over(self):
        """Check if game has ended."""
        return len(self.get_legal_moves()) == 0 or self.move_count >= self.max_moves
    
    def get_score(self):
        """
        Evaluate current position from player 1's perspective.
        Higher score favors player 1, lower score favors player 2.
        """
        if self.is_game_over():
            return self.final_evaluation()
        return self.heuristic_evaluation()
    
    def final_evaluation(self):
        """Calculate final score when game is over."""
        score1 = self._calculate_player_score(1)
        score2 = self._calculate_player_score(2)
        return score1 - score2
    
    def heuristic_evaluation(self):
        """Evaluate position during game."""
        score1 = self._calculate_player_score(1)
        score2 = self._calculate_player_score(2)
        return score1 - score2
    
    def _calculate_player_score(self, player):
        """Calculate score for a specific player."""
        score = 0
        
        # Count pieces
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == player:
                    score += 1
                    # Center control bonus
                    if 2 <= row <= 3 and 2 <= col <= 3:
                        score += 2
        
        # Check for patterns (lines of 2, 3, 4)
        score += self._count_patterns(player, 2) * 3
        score += self._count_patterns(player, 3) * 10
        score += self._count_patterns(player, 4) * 50
        
        return score
    
    def _count_patterns(self, player, length):
        """Count patterns of given length for player."""
        count = 0
        
        # Horizontal
        for row in range(self.size):
            for col in range(self.size - length + 1):
                if all(self.board[row][col + i] == player for i in range(length)):
                    count += 1
        
        # Vertical
        for row in range(self.size - length + 1):
            for col in range(self.size):
                if all(self.board[row + i][col] == player for i in range(length)):
                    count += 1
        
        # Diagonal
        for row in range(self.size - length + 1):
            for col in range(self.size - length + 1):
                if all(self.board[row + i][col + i] == player for i in range(length)):
                    count += 1
                if all(self.board[row + i][col + length - 1 - i] == player for i in range(length)):
                    count += 1
        
        return count
    
    def whose_turn(self):
        """Return which player's turn it is."""
        return self.current_player


# Global counter for nodes evaluated
nodes_evaluated = 0


def minimax_alpha_beta(position, depth, alpha, beta, maximizing_player):
    """
    Minimax search with alpha-beta pruning.
    
    Args:
        position: Current GameState
        depth: Remaining search depth
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        maximizing_player: True if maximizing, False if minimizing
    
    Returns:
        (best_score, best_move) tuple
    """
    global nodes_evaluated
    nodes_evaluated += 1
    
    if depth == 0 or position.is_game_over():
        return position.get_score(), None
    
    legal_moves = position.get_legal_moves()
    
    if not legal_moves:
        return position.get_score(), None
    
    best_move = None
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            new_position = position.make_move(move)
            eval_score, _ = minimax_alpha_beta(new_position, depth - 1, alpha, beta, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            new_position = position.make_move(move)
            eval_score, _ = minimax_alpha_beta(new_position, depth - 1, alpha, beta, True)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        
        return min_eval, best_move


def find_best_move(position, depth):
    """
    Find the best move for the current position.
    
    Args:
        position: Current GameState
        depth: Search depth
    
    Returns:
        best_move: The best move found (row, col) tuple
    """
    global nodes_evaluated
    nodes_evaluated = 0
    
    maximizing = (position.whose_turn() == 1)
    _, best_move = minimax_alpha_beta(position, depth, float('-inf'), float('inf'), maximizing)
    
    return best_move


def get_nodes_evaluated():
    """Return the number of nodes evaluated in the last search."""
    return nodes_evaluated


if __name__ == "__main__":
    # Simple test
    game = GameState()
    print("Starting position:")
    for row in game.board:
        print(row)
    
    print("\nFinding best move at depth 4...")
    best_move = find_best_move(game, 4)
    print(f"Best move: {best_move}")
    print(f"Nodes evaluated: {get_nodes_evaluated()}")