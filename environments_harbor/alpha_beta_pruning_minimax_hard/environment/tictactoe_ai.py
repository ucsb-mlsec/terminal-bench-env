#!/usr/bin/env python3

import json
import copy

class TicTacToeBoard:
    """Represents a Tic-Tac-Toe board state"""
    
    def __init__(self, board=None):
        if board is None:
            self.board = [[' ' for _ in range(3)] for _ in range(3)]
        else:
            self.board = copy.deepcopy(board)
    
    def make_move(self, row, col, player):
        """Create a new board with the move applied"""
        new_board = TicTacToeBoard(self.board)
        new_board.board[row][col] = player
        return new_board
    
    def is_valid_move(self, row, col):
        """Check if a move is valid"""
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == ' '
    
    def get_available_moves(self):
        """Get all empty cells as list of (row, col) tuples"""
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    moves.append((row, col))
        return moves
    
    def check_winner(self):
        """Check if there's a winner. Returns 'X', 'O', or None"""
        # Check rows
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] != ' ':
                return self.board[row][0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def is_game_over(self):
        """Check if game is finished (win or draw)"""
        if self.check_winner() is not None:
            return True
        return len(self.get_available_moves()) == 0
    
    def __str__(self):
        """String representation for debugging"""
        lines = []
        for row in self.board:
            lines.append('|'.join(row))
        return '\n-----\n'.join(lines)


def evaluate_position(board):
    """
    Evaluate the board position.
    Returns +10 if X wins, -10 if O wins, 0 otherwise
    """
    winner = board.check_winner()
    if winner == 'X':
        return 10
    elif winner == 'O':
        return -10
    else:
        return 0


def minimax(board, player, is_maximizing):
    """
    SLOW minimax implementation without any optimizations.
    Explores ALL possible game states.
    
    Args:
        board: Current board state
        player: Current player ('X' or 'O')
        is_maximizing: True if maximizing player, False if minimizing
    
    Returns:
        Best score for the current player
    """
    # Check if game is over
    if board.is_game_over():
        return evaluate_position(board)
    
    available_moves = board.get_available_moves()
    
    if is_maximizing:
        # Maximizing player (X)
        best_score = float('-inf')
        for row, col in available_moves:
            new_board = board.make_move(row, col, player)
            next_player = 'O' if player == 'X' else 'X'
            score = minimax(new_board, next_player, False)
            best_score = max(best_score, score)
        return best_score
    else:
        # Minimizing player (O)
        best_score = float('inf')
        for row, col in available_moves:
            new_board = board.make_move(row, col, player)
            next_player = 'O' if player == 'X' else 'X'
            score = minimax(new_board, next_player, True)
            best_score = min(best_score, score)
        return best_score


def find_best_move(board, player):
    """
    Find the best move for the given player using minimax.
    This is SLOW because it evaluates all moves completely without pruning.
    
    Args:
        board: TicTacToeBoard instance
        player: Current player ('X' or 'O')
    
    Returns:
        Tuple (row, col) representing the best move
    """
    available_moves = board.get_available_moves()
    
    if not available_moves:
        return None
    
    best_move = None
    is_maximizing = (player == 'X')
    
    if is_maximizing:
        best_score = float('-inf')
        for row, col in available_moves:
            new_board = board.make_move(row, col, player)
            next_player = 'O'
            score = minimax(new_board, next_player, False)
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
    else:
        best_score = float('inf')
        for row, col in available_moves:
            new_board = board.make_move(row, col, player)
            next_player = 'X'
            score = minimax(new_board, next_player, True)
            
            if score < best_score:
                best_score = score
                best_move = (row, col)
    
    return best_move


def load_position(filename):
    """
    Load a position from a JSON file.
    
    Args:
        filename: Path to JSON file
    
    Returns:
        Tuple (board, next_player) where board is a TicTacToeBoard instance
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    board = TicTacToeBoard(data['board'])
    next_player = data['next_player']
    
    return board, next_player


if __name__ == '__main__':
    # Test with a simple position
    test_board = TicTacToeBoard([
        ['X', 'O', ' '],
        [' ', 'X', ' '],
        ['O', ' ', ' ']
    ])
    
    print("Test board:")
    print(test_board)
    print()
    
    print("Finding best move for X...")
    best_move = find_best_move(test_board, 'X')
    print(f"Best move: {best_move}")