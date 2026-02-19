#!/usr/bin/env python3

import random
from typing import Optional, Tuple, List, Set


class GoEngine:
    """A basic Go game engine with board management and move validation."""
    
    def __init__(self, board_size: int = 19):
        """Initialize the Go engine with specified board size."""
        self.board_size = board_size
        self.board = None
        self.current_player = 'B'
        self.valid_sizes = [9, 13, 19]
        self.clear_board()
    
    def set_board_size(self, size: int) -> bool:
        """Set the board size. Only 9, 13, or 19 are valid."""
        if size not in self.valid_sizes:
            return False
        self.board_size = size
        self.clear_board()
        return True
    
    def clear_board(self):
        """Reset the board to empty state."""
        self.board = [[None for _ in range(self.board_size)] 
                      for _ in range(self.board_size)]
        self.current_player = 'B'
    
    def vertex_to_coords(self, vertex: str) -> Optional[Tuple[int, int]]:
        """
        Convert vertex notation (e.g., 'C3') to board coordinates.
        Letters A-T (skipping I), numbers 1-19.
        Returns (row, col) or None if invalid.
        """
        if not vertex or len(vertex) < 2:
            return None
        
        vertex = vertex.upper()
        col_letter = vertex[0]
        row_str = vertex[1:]
        
        # Handle column (A-T, skip I)
        if col_letter < 'A' or col_letter > 'T':
            return None
        
        # Skip I in the column letters
        if col_letter >= 'I':
            col = ord(col_letter) - ord('A') - 1
        else:
            col = ord(col_letter) - ord('A')
        
        # Handle row number
        try:
            row_num = int(row_str)
        except ValueError:
            return None
        
        if row_num < 1 or row_num > self.board_size:
            return None
        
        # Convert to 0-indexed, flip row (GTP uses bottom-left origin)
        row = self.board_size - row_num
        
        if col >= self.board_size:
            return None
        
        return (row, col)
    
    def coords_to_vertex(self, row: int, col: int) -> str:
        """Convert board coordinates to vertex notation."""
        # Convert column to letter (skip I)
        if col >= 8:
            col_letter = chr(ord('A') + col + 1)
        else:
            col_letter = chr(ord('A') + col)
        
        # Convert row (flip back from 0-indexed)
        row_num = self.board_size - row
        
        return f"{col_letter}{row_num}"
    
    def is_valid_move(self, color: str, vertex: str) -> Tuple[bool, str]:
        """
        Check if a move is legal.
        Returns (is_valid, error_message).
        """
        color = color.upper()
        if color not in ['B', 'W']:
            return False, "invalid color"
        
        if vertex.upper() == 'PASS':
            return True, ""
        
        coords = self.vertex_to_coords(vertex)
        if coords is None:
            return False, "invalid vertex"
        
        row, col = coords
        
        if self.board[row][col] is not None:
            return False, "illegal move"
        
        return True, ""
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return list of adjacent positions (up, down, left, right)."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group (connected stones of same color)."""
        color = self.board[row][col]
        if color is None:
            return set()
        
        group = set()
        to_check = [(row, col)]
        
        while to_check:
            r, c = to_check.pop()
            if (r, c) in group:
                continue
            
            if self.board[r][c] == color:
                group.add((r, c))
                for nr, nc in self.get_neighbors(r, c):
                    if (nr, nc) not in group:
                        to_check.append((nr, nc))
        
        return group
    
    def has_liberties(self, row: int, col: int) -> bool:
        """Check if the stone/group at position has any liberties."""
        group = self.get_group(row, col)
        
        for r, c in group:
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr][nc] is None:
                    return True
        
        return False
    
    def remove_captured_stones(self, color: str) -> int:
        """
        Remove captured opponent stones (stones with no liberties).
        Returns the number of stones removed.
        """
        opponent = 'W' if color == 'B' else 'B'
        captured = 0
        checked = set()
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == opponent and (row, col) not in checked:
                    group = self.get_group(row, col)
                    checked.update(group)
                    
                    if not self.has_liberties(row, col):
                        # Remove the captured group
                        for r, c in group:
                            self.board[r][c] = None
                            captured += 1
        
        return captured
    
    def play_move(self, color: str, vertex: str) -> Tuple[bool, str]:
        """
        Place a stone on the board.
        Returns (success, error_message).
        """
        color = color.upper()
        vertex = vertex.upper()
        
        if vertex == 'PASS':
            return True, ""
        
        is_valid, error = self.is_valid_move(color, vertex)
        if not is_valid:
            return False, error
        
        coords = self.vertex_to_coords(vertex)
        if coords is None:
            return False, "invalid vertex"
        
        row, col = coords
        self.board[row][col] = color
        
        # Remove captured opponent stones
        self.remove_captured_stones(color)
        
        # Check for suicide (simplified - just check if own stone has liberties)
        if not self.has_liberties(row, col):
            # In basic implementation, we could reject suicide
            # For now, we'll allow it (some rulesets allow suicide if it captures)
            pass
        
        return True, ""
    
    def generate_move(self, color: str) -> str:
        """Generate a random legal move for the specified color."""
        color = color.upper()
        legal_moves = []
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] is None:
                    vertex = self.coords_to_vertex(row, col)
                    is_valid, _ = self.is_valid_move(color, vertex)
                    if is_valid:
                        legal_moves.append(vertex)
        
        if not legal_moves:
            return 'PASS'
        
        return random.choice(legal_moves)
    
    def get_board_state(self) -> str:
        """Return a string representation of the current board."""
        lines = []
        
        # Column headers
        col_headers = '   '
        for col in range(self.board_size):
            if col >= 8:
                col_headers += chr(ord('A') + col + 1) + ' '
            else:
                col_headers += chr(ord('A') + col) + ' '
        lines.append(col_headers)
        
        # Board rows
        for row in range(self.board_size):
            row_num = self.board_size - row
            row_str = f"{row_num:2d} "
            
            for col in range(self.board_size):
                stone = self.board[row][col]
                if stone == 'B':
                    row_str += 'X '
                elif stone == 'W':
                    row_str += 'O '
                else:
                    row_str += '. '
            
            row_str += f"{row_num:2d}"
            lines.append(row_str)
        
        # Bottom column headers
        lines.append(col_headers)
        
        return '\n'.join(lines)