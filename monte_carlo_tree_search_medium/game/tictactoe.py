#!/usr/bin/env python3

class TicTacToe:
    """
    A complete Tic-Tac-Toe game engine.
    Board representation: 3x3 grid with 0 (empty), 1 (X), 2 (O)
    """
    
    def __init__(self):
        """Initialize an empty 3x3 board"""
        self.board = [[0, 0, 0] for _ in range(3)]
    
    def make_move(self, row, col, player):
        """
        Place a move on the board.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            player: Player number (1 for X, 2 for O)
        
        Returns:
            True if move was valid and placed, False otherwise
        """
        if not self.is_valid_move(row, col):
            return False
        
        if player not in [1, 2]:
            return False
        
        self.board[row][col] = player
        return True
    
    def is_valid_move(self, row, col):
        """
        Check if a move is valid (in bounds and position is empty).
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
        
        Returns:
            True if move is valid, False otherwise
        """
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        
        return self.board[row][col] == 0
    
    def check_winner(self):
        """
        Check the current game state.
        
        Returns:
            1 if X wins
            2 if O wins
            0 if game is ongoing or draw
        """
        # Check for player 1 (X) win
        if check_win_condition(self.board, 1):
            return 1
        
        # Check for player 2 (O) win
        if check_win_condition(self.board, 2):
            return 2
        
        # No winner yet or draw
        return 0
    
    def is_game_over(self):
        """
        Check if the game has ended (win or draw).
        
        Returns:
            True if game is over, False otherwise
        """
        if self.check_winner() != 0:
            return True
        
        return is_board_full(self.board)
    
    def get_board(self):
        """
        Get the current board state.
        
        Returns:
            3x3 list of lists representing the board
        """
        return [row[:] for row in self.board]
    
    def get_available_moves(self):
        """
        Get all available moves on the board.
        
        Returns:
            List of (row, col) tuples for empty positions
        """
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 0:
                    moves.append((row, col))
        return moves
    
    def reset(self):
        """Clear the board to initial empty state"""
        self.board = [[0, 0, 0] for _ in range(3)]


def print_board(board):
    """
    Display the board in a readable format.
    
    Args:
        board: 3x3 list of lists with values 0, 1, or 2
    """
    symbols = {0: '.', 1: 'X', 2: 'O'}
    
    print("\n  0 1 2")
    for i, row in enumerate(board):
        print(f"{i} {' '.join(symbols[cell] for cell in row)}")
    print()


def check_win_condition(board, player):
    """
    Check if the given player has won.
    
    Args:
        board: 3x3 list of lists
        player: Player number (1 or 2)
    
    Returns:
        True if player has won, False otherwise
    """
    # Check rows
    for row in range(3):
        if all(board[row][col] == player for col in range(3)):
            return True
    
    # Check columns
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    
    # Check diagonal (top-left to bottom-right)
    if all(board[i][i] == player for i in range(3)):
        return True
    
    # Check anti-diagonal (top-right to bottom-left)
    if all(board[i][2-i] == player for i in range(3)):
        return True
    
    return False


def is_board_full(board):
    """
    Check if the board is completely filled.
    
    Args:
        board: 3x3 list of lists
    
    Returns:
        True if no empty spaces remain, False otherwise
    """
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                return False
    return True


# Alias for compatibility
GameEngine = TicTacToe


if __name__ == "__main__":
    # Example usage
    game = TicTacToe()
    
    print("New game started!")
    print_board(game.get_board())
    
    # Simulate a few moves
    game.make_move(0, 0, 1)  # X plays top-left
    print("X plays (0, 0)")
    print_board(game.get_board())
    
    game.make_move(1, 1, 2)  # O plays center
    print("O plays (1, 1)")
    print_board(game.get_board())
    
    game.make_move(0, 1, 1)  # X plays top-center
    print("X plays (0, 1)")
    print_board(game.get_board())
    
    game.make_move(2, 2, 2)  # O plays bottom-right
    print("O plays (2, 2)")
    print_board(game.get_board())
    
    game.make_move(0, 2, 1)  # X plays top-right
    print("X plays (0, 2)")
    print_board(game.get_board())
    
    winner = game.check_winner()
    if winner:
        print(f"Player {'X' if winner == 1 else 'O'} wins!")
    elif game.is_game_over():
        print("Game is a draw!")
    
    print(f"Available moves: {game.get_available_moves()}")