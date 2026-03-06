#!/usr/bin/env python3

class GomokuGame:
    """Gomoku (Five-in-a-Row) game engine for 15x15 board."""
    
    BOARD_SIZE = 15
    BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
    WIN_LENGTH = 5
    
    def __init__(self, board_state=None):
        """Initialize game with optional board state.
        
        Args:
            board_state: 225-character string ('X', 'O', '.') or None for empty board
        """
        if board_state is None:
            self.board = ['.'] * self.BOARD_CELLS
        else:
            if len(board_state) != self.BOARD_CELLS:
                raise ValueError(f"Board state must be {self.BOARD_CELLS} characters")
            self.board = list(board_state)
    
    def get_legal_moves(self):
        """Returns list of valid position integers (0-224) for empty intersections."""
        return [i for i in range(self.BOARD_CELLS) if self.board[i] == '.']
    
    def make_move(self, position, player):
        """Apply move and return new board state.
        
        Args:
            position: integer 0-224
            player: 1 or 2
            
        Returns:
            New board state as 225-character string
            
        Raises:
            ValueError: if move is illegal
        """
        if position < 0 or position >= self.BOARD_CELLS:
            raise ValueError(f"Position {position} out of bounds")
        
        if self.board[position] != '.':
            raise ValueError(f"Position {position} already occupied")
        
        if player not in [1, 2]:
            raise ValueError(f"Player must be 1 or 2, got {player}")
        
        # Create new board state
        new_board = self.board.copy()
        new_board[position] = 'X' if player == 1 else 'O'
        self.board = new_board
        
        return self.get_board_string()
    
    def check_winner(self):
        """Check for winner.
        
        Returns:
            1 if player 1 wins, 2 if player 2 wins, None if no winner
        """
        # Check all possible winning lines for both players
        for player, symbol in [(1, 'X'), (2, 'O')]:
            if self._has_five_in_row(symbol):
                return player
        return None
    
    def _has_five_in_row(self, symbol):
        """Check if given symbol has 5 in a row."""
        # Check horizontal
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
                if all(self.board[row * self.BOARD_SIZE + col + i] == symbol 
                       for i in range(self.WIN_LENGTH)):
                    return True
        
        # Check vertical
        for row in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
            for col in range(self.BOARD_SIZE):
                if all(self.board[(row + i) * self.BOARD_SIZE + col] == symbol 
                       for i in range(self.WIN_LENGTH)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
            for col in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
                if all(self.board[(row + i) * self.BOARD_SIZE + col + i] == symbol 
                       for i in range(self.WIN_LENGTH)):
                    return True
        
        # Check anti-diagonal (top-right to bottom-left)
        for row in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
            for col in range(self.WIN_LENGTH - 1, self.BOARD_SIZE):
                if all(self.board[(row + i) * self.BOARD_SIZE + col - i] == symbol 
                       for i in range(self.WIN_LENGTH)):
                    return True
        
        return False
    
    def get_board_string(self):
        """Returns current board as 225-character string."""
        return ''.join(self.board)
    
    def copy(self):
        """Create a copy of the current game state."""
        return GomokuGame(self.get_board_string())
    
    def position_to_coords(self, position):
        """Convert position (0-224) to (row, col) coordinates."""
        return position // self.BOARD_SIZE, position % self.BOARD_SIZE
    
    def coords_to_position(self, row, col):
        """Convert (row, col) coordinates to position (0-224)."""
        return row * self.BOARD_SIZE + col
    
    def get_stone(self, position):
        """Get stone at position ('X', 'O', or '.')."""
        if position < 0 or position >= self.BOARD_CELLS:
            return None
        return self.board[position]
    
    def is_empty(self, position):
        """Check if position is empty."""
        return 0 <= position < self.BOARD_CELLS and self.board[position] == '.'
    
    def get_player_symbol(self, player):
        """Get symbol for player (1='X', 2='O')."""
        return 'X' if player == 1 else 'O'
    
    def get_opponent(self, player):
        """Get opponent player number."""
        return 2 if player == 1 else 1
    
    def is_game_over(self):
        """Check if game is over (winner exists or board full)."""
        if self.check_winner() is not None:
            return True
        return len(self.get_legal_moves()) == 0
    
    def display(self):
        """Display board in human-readable format."""
        print("   " + " ".join(f"{i:2d}" for i in range(self.BOARD_SIZE)))
        for row in range(self.BOARD_SIZE):
            print(f"{row:2d} ", end="")
            for col in range(self.BOARD_SIZE):
                pos = self.coords_to_position(row, col)
                print(f" {self.board[pos]} ", end="")
            print()
        print()