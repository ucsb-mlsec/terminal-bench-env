#!/usr/bin/env python3

import sys

# Constants for board state
EMPTY = 0
BLACK = 1
WHITE = 2

class Board:
    """Manages a 9x9 Go board state."""
    
    def __init__(self):
        """Initialize an empty 9x9 board."""
        self.size = 9
        self.board = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
    
    def get(self, row, col):
        """Get the stone at position (row, col)."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.board[row][col]
        return None
    
    def set(self, row, col, color):
        """Place a stone at position (row, col)."""
        if 0 <= row < self.size and 0 <= col < self.size:
            self.board[row][col] = color
    
    def is_empty(self, row, col):
        """Check if position (row, col) is empty."""
        return self.get(row, col) == EMPTY
    
    def clear(self):
        """Clear the board."""
        self.board = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
    
    def copy(self):
        """Create a copy of the board."""
        new_board = Board()
        for row in range(self.size):
            for col in range(self.size):
                new_board.board[row][col] = self.board[row][col]
        return new_board
    
    def get_neighbors(self, row, col):
        """Get list of (row, col) tuples for valid neighbors."""
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append((r, c))
        return neighbors
    
    def __eq__(self, other):
        """Check if two boards are equal."""
        if not isinstance(other, Board):
            return False
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != other.board[row][col]:
                    return False
        return True


class GTPEngine:
    """GTP Engine for processing Go Text Protocol commands."""
    
    def __init__(self):
        """Initialize the GTP engine."""
        self.board = Board()
        self.history = []
    
    def vertex_to_coords(self, vertex):
        """
        Convert GTP vertex notation to board coordinates.
        E.g., 'C3' -> (2, 2), 'A1' -> (0, 0)
        Columns: A-H, J (excluding I)
        Rows: 1-9
        """
        if not vertex or len(vertex) < 2:
            return None
        
        col_letter = vertex[0].upper()
        row_str = vertex[1:]
        
        # Map column letters to indices (A=0, B=1, ..., H=7, J=8)
        col_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}
        
        if col_letter not in col_map:
            return None
        
        col = col_map[col_letter]
        
        try:
            row = int(row_str) - 1  # Convert to 0-indexed
            if row < 0 or row >= 9:
                return None
        except ValueError:
            return None
        
        return (row, col)
    
    def parse_color(self, color_str):
        """Parse color string to constant."""
        color_upper = color_str.upper()
        if color_upper == 'B' or color_upper == 'BLACK':
            return BLACK
        elif color_upper == 'W' or color_upper == 'WHITE':
            return WHITE
        return None
    
    def play(self, color_str, vertex):
        """
        Process a play command WITHOUT validation.
        Places stone directly on the board.
        """
        color = self.parse_color(color_str)
        if color is None:
            return False
        
        coords = self.vertex_to_coords(vertex)
        if coords is None:
            return False
        
        row, col = coords
        
        # No validation - just place the stone
        self.board.set(row, col, color)
        self.history.append(self.board.copy())
        return True
    
    def process_command(self, command):
        """Process a GTP command string."""
        parts = command.strip().split()
        if not parts:
            return False
        
        cmd = parts[0].lower()
        
        if cmd == 'play':
            if len(parts) >= 3:
                color = parts[1]
                vertex = parts[2]
                return self.play(color, vertex)
        elif cmd == 'clear_board':
            self.board.clear()
            self.history = []
            return True
        
        return False
    
    def get_board_state(self):
        """Get current board state."""
        return self.board.copy()


def main():
    """Main function for running the GTP engine."""
    engine = GTPEngine()
    
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        engine.process_command(line)


if __name__ == '__main__':
    main()