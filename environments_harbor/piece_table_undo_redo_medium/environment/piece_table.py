#!/usr/bin/env python3

class PieceTable:
    """
    Piece Table implementation for a text editor.
    
    The piece table uses two buffers:
    - original: The initial text that never changes
    - add_buffer: New text added through insert operations
    
    Pieces are descriptors that point to segments in these buffers.
    Each piece has:
    - source: 'original' or 'add' (which buffer to read from)
    - start: starting position in that buffer
    - length: number of characters in this piece
    
    Operations modify the piece descriptors, not the underlying buffers.
    This makes undo/redo efficient.
    """
    
    def __init__(self, original_text=""):
        """Initialize the piece table with original text."""
        self.original = original_text
        self.add_buffer = ""
        
        # Start with a single piece covering all original text
        if original_text:
            self.pieces = [{'source': 'original', 'start': 0, 'length': len(original_text)}]
        else:
            self.pieces = []
        
        # History tracking for undo/redo
        self.history = []  # Stack of previous states
        self.redo_stack = []  # Stack for redo operations
    
    def _save_state(self):
        """Save current state to history for undo."""
        # Deep copy the pieces list
        state = {
            'pieces': [p.copy() for p in self.pieces],
            'add_buffer': self.add_buffer
        }
        self.history.append(state)
        # Clear redo stack when new operation is performed
        self.redo_stack = []
    
    def insert(self, position, text):
        """Insert text at the given position."""
        if not text:
            return
        
        # Save current state for undo
        self._save_state()
        
        # Add new text to add_buffer
        add_start = len(self.add_buffer)
        self.add_buffer += text
        
        # Find which piece contains the position
        current_pos = 0
        new_pieces = []
        inserted = False
        
        for piece in self.pieces:
            piece_end = current_pos + piece['length']
            
            if current_pos <= position <= piece_end and not inserted:
                # Split this piece
                offset = position - current_pos
                
                # First part (before insertion point)
                if offset > 0:
                    new_pieces.append({
                        'source': piece['source'],
                        'start': piece['start'],
                        'length': offset
                    })
                
                # New piece for inserted text
                new_pieces.append({
                    'source': 'add',
                    'start': add_start,
                    'length': len(text)
                })
                
                # Second part (after insertion point)
                remaining = piece['length'] - offset
                if remaining > 0:
                    new_pieces.append({
                        'source': piece['source'],
                        'start': piece['start'] + offset,
                        'length': remaining
                    })
                
                inserted = True
            else:
                new_pieces.append(piece)
            
            current_pos = piece_end
        
        # If position is at the end, just append
        if not inserted:
            new_pieces.append({
                'source': 'add',
                'start': add_start,
                'length': len(text)
            })
        
        self.pieces = new_pieces
    
    def delete(self, position, length):
        """Delete text starting at position for the given length."""
        if length <= 0:
            return
        
        # Save current state for undo
        self._save_state()
        
        delete_end = position + length
        current_pos = 0
        new_pieces = []
        
        for piece in self.pieces:
            piece_start = current_pos
            piece_end = current_pos + piece['length']
            
            # Piece is completely before deletion range
            if piece_end <= position:
                new_pieces.append(piece)
            # Piece is completely after deletion range
            elif piece_start >= delete_end:
                new_pieces.append(piece)
            # Piece overlaps with deletion range
            else:
                # Keep part before deletion
                if piece_start < position:
                    keep_length = position - piece_start
                    new_pieces.append({
                        'source': piece['source'],
                        'start': piece['start'],
                        'length': keep_length
                    })
                
                # Keep part after deletion
                if piece_end > delete_end:
                    skip = delete_end - piece_start
                    keep_length = piece_end - delete_end
                    new_pieces.append({
                        'source': piece['source'],
                        'start': piece['start'] + skip,
                        'length': keep_length
                    })
            
            current_pos = piece_end
        
        self.pieces = new_pieces
    
    def undo(self):
        """Undo the most recent operation."""
        # TODO: Implement this
        if not self.history:
            return False
        
        # Save current state to redo stack
        current_state = {
            'pieces': [p.copy() for p in self.pieces],
            'add_buffer': self.add_buffer
        }
        self.redo_stack.append(current_state)
        
        # Restore previous state
        previous_state = self.history.pop()
        self.pieces = previous_state['pieces']
        self.add_buffer = previous_state['add_buffer']
        
        return True
    
    def redo(self):
        """Redo an undone operation."""
        # TODO: Implement this
        if not self.redo_stack:
            return False
        
        # Save current state to history
        current_state = {
            'pieces': [p.copy() for p in self.pieces],
            'add_buffer': self.add_buffer
        }
        self.history.append(current_state)
        
        # Restore redo state
        redo_state = self.redo_stack.pop()
        self.pieces = redo_state['pieces']
        self.add_buffer = redo_state['add_buffer']
        
        return True
    
    def get_text(self):
        """Reconstruct and return the current text from the piece table."""
        result = []
        
        for piece in self.pieces:
            if piece['source'] == 'original':
                result.append(self.original[piece['start']:piece['start'] + piece['length']])
            else:  # source == 'add'
                result.append(self.add_buffer[piece['start']:piece['start'] + piece['length']])
        
        return ''.join(result)