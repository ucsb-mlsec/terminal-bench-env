#!/usr/bin/env python3
"""
Chess Position Generator
Generates random chess positions in FEN (Forsyth-Edwards Notation) format.

FEN Format: [piece placement] [active color] [castling] [en passant] [halfmove] [fullmove]
Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
"""

import random


def generate_piece_placement():
    """
    Generate random piece placement for the 8 ranks of a chess board.
    Each rank is represented as pieces and empty squares (as digits 1-8).
    
    Returns:
        str: Piece placement string with ranks separated by '/'
    """
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    ranks = []
    
    # Ensure each side has exactly one king
    has_white_king = False
    has_black_king = False
    
    for rank_idx in range(8):
        rank = []
        empty_count = 0
        
        for file_idx in range(8):
            # Randomly decide if square is empty (60% chance)
            if random.random() < 0.6:
                empty_count += 1
            else:
                # Add accumulated empty squares
                if empty_count > 0:
                    rank.append(str(empty_count))
                    empty_count = 0
                
                # Select a random piece
                available_pieces = pieces.copy()
                
                # Ensure we place kings
                if rank_idx == 7 and file_idx >= 4 and not has_white_king:
                    piece = 'K'
                    has_white_king = True
                elif rank_idx == 0 and file_idx >= 4 and not has_black_king:
                    piece = 'k'
                    has_black_king = True
                else:
                    # Limit king placement to one per side
                    if has_white_king:
                        available_pieces = [p for p in available_pieces if p != 'K']
                    if has_black_king:
                        available_pieces = [p for p in available_pieces if p != 'k']
                    
                    piece = random.choice(available_pieces)
                    
                    if piece == 'K':
                        has_white_king = True
                    elif piece == 'k':
                        has_black_king = True
                
                rank.append(piece)
        
        # Add remaining empty squares
        if empty_count > 0:
            rank.append(str(empty_count))
        
        # If rank is completely empty, represent as '8'
        if not rank:
            rank.append('8')
        
        ranks.append(''.join(rank))
    
    return '/'.join(ranks)


def generate_active_color():
    """Generate active color (w for white, b for black)."""
    return random.choice(['w', 'b'])


def generate_castling_rights():
    """Generate castling availability (KQkq or combinations or -)."""
    rights = []
    for castle in ['K', 'Q', 'k', 'q']:
        if random.random() < 0.3:
            rights.append(castle)
    return ''.join(rights) if rights else '-'


def generate_en_passant():
    """Generate en passant target square (e.g., e3 or -)."""
    if random.random() < 0.1:  # 10% chance of en passant
        file = random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        rank = random.choice(['3', '6'])
        return f"{file}{rank}"
    return '-'


def generate_halfmove_clock():
    """Generate halfmove clock (0-50 for fifty-move rule)."""
    return str(random.randint(0, 50))


def generate_fullmove_number():
    """Generate fullmove number (starts at 1, typically 1-100 for testing)."""
    return str(random.randint(1, 100))


def generate_random_position():
    """
    Generate a complete random chess position in FEN notation.
    
    Returns:
        str: A valid FEN position string
    """
    components = [
        generate_piece_placement(),
        generate_active_color(),
        generate_castling_rights(),
        generate_en_passant(),
        generate_halfmove_clock(),
        generate_fullmove_number()
    ]
    return ' '.join(components)


def generate_position_set(count: int):
    """
    Generate a set of random chess positions.
    
    Args:
        count: Number of positions to generate
        
    Returns:
        list: List of FEN position strings
    """
    if count <= 0:
        raise ValueError("Count must be positive")
    
    positions = []
    for _ in range(count):
        positions.append(generate_random_position())
    
    return positions


if __name__ == '__main__':
    print("Chess Position Generator - Sample Output")
    print("=" * 80)
    print("\nGenerating 10 random chess positions in FEN notation:\n")
    
    for i, position in enumerate(generate_position_set(10), 1):
        print(f"{i}. {position}")