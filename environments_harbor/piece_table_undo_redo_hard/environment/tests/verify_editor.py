#!/usr/bin/env python3

import sys
import os

# Add solution directory to path
sys.path.insert(0, '/workspace/solution')

def find_editor_class():
    """Try to import and find the editor class from solution.editor"""
    try:
        import editor as editor_module
    except ImportError as e:
        print(f"FATAL ERROR: Could not import solution.editor module: {e}")
        sys.exit(1)
    
    # Try common class names
    possible_names = ['TextEditor', 'Editor', 'PieceTableEditor', 'DocumentEditor', 'PieceTable']
    
    for name in possible_names:
        if hasattr(editor_module, name):
            return getattr(editor_module, name)
    
    # If none found, try to get any class from the module
    for attr_name in dir(editor_module):
        attr = getattr(editor_module, attr_name)
        if isinstance(attr, type) and not attr_name.startswith('_'):
            return attr
    
    print("FATAL ERROR: Could not find editor class in solution.editor module")
    print("Looked for classes:", possible_names)
    sys.exit(1)

# Get the editor class
EditorClass = find_editor_class()

def test_basic_insert():
    """Test basic insert operation"""
    try:
        editor = EditorClass()
        editor.insert(0, "Hello")
        result = editor.get_text()
        if result == "Hello":
            print("✓ test_basic_insert: PASS")
            return True
        else:
            print(f"✗ test_basic_insert: FAIL - Expected 'Hello', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_basic_insert: FAIL - Exception: {e}")
        return False

def test_multiple_inserts():
    """Test multiple insert operations"""
    try:
        editor = EditorClass()
        editor.insert(0, "Hello")
        editor.insert(5, " World")
        result = editor.get_text()
        if result == "Hello World":
            print("✓ test_multiple_inserts: PASS")
            return True
        else:
            print(f"✗ test_multiple_inserts: FAIL - Expected 'Hello World', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_multiple_inserts: FAIL - Exception: {e}")
        return False

def test_basic_delete():
    """Test basic delete operation"""
    try:
        editor = EditorClass()
        editor.insert(0, "Hello World")
        editor.delete(5, 6)
        result = editor.get_text()
        if result == "Hello":
            print("✓ test_basic_delete: PASS")
            return True
        else:
            print(f"✗ test_basic_delete: FAIL - Expected 'Hello', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_basic_delete: FAIL - Exception: {e}")
        return False

def test_simple_undo():
    """Test simple undo operation"""
    try:
        editor = EditorClass()
        editor.insert(0, "Hello")
        editor.insert(5, " World")
        editor.undo()
        result = editor.get_text()
        if result == "Hello":
            print("✓ test_simple_undo: PASS")
            return True
        else:
            print(f"✗ test_simple_undo: FAIL - Expected 'Hello', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_simple_undo: FAIL - Exception: {e}")
        return False

def test_simple_redo():
    """Test simple redo operation"""
    try:
        editor = EditorClass()
        editor.insert(0, "Hello")
        editor.insert(5, " World")
        editor.undo()
        editor.redo()
        result = editor.get_text()
        if result == "Hello World":
            print("✓ test_simple_redo: PASS")
            return True
        else:
            print(f"✗ test_simple_redo: FAIL - Expected 'Hello World', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_simple_redo: FAIL - Exception: {e}")
        return False

def test_multiple_undo_redo():
    """Test multiple undo and redo operations"""
    try:
        editor = EditorClass()
        editor.insert(0, "A")
        editor.insert(1, "B")
        editor.insert(2, "C")
        
        editor.undo()
        editor.undo()
        result1 = editor.get_text()
        
        editor.redo()
        editor.redo()
        result2 = editor.get_text()
        
        if result1 == "A" and result2 == "ABC":
            print("✓ test_multiple_undo_redo: PASS")
            return True
        else:
            print(f"✗ test_multiple_undo_redo: FAIL - After 2 undos: '{result1}' (expected 'A'), after 2 redos: '{result2}' (expected 'ABC')")
            return False
    except Exception as e:
        print(f"✗ test_multiple_undo_redo: FAIL - Exception: {e}")
        return False

def test_edit_after_undo_clears_redo():
    """Test that editing after undo clears redo history"""
    try:
        editor = EditorClass()
        editor.insert(0, "A")
        editor.insert(1, "B")
        editor.undo()
        editor.insert(1, "C")
        editor.redo()  # Should have no effect
        result = editor.get_text()
        if result == "AC":
            print("✓ test_edit_after_undo_clears_redo: PASS")
            return True
        else:
            print(f"✗ test_edit_after_undo_clears_redo: FAIL - Expected 'AC', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_edit_after_undo_clears_redo: FAIL - Exception: {e}")
        return False

def test_undo_on_empty():
    """Test undo on empty history"""
    try:
        editor = EditorClass()
        editor.undo()  # Should not crash
        result = editor.get_text()
        if result == "":
            print("✓ test_undo_on_empty: PASS")
            return True
        else:
            print(f"✗ test_undo_on_empty: FAIL - Expected empty string, got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_undo_on_empty: FAIL - Exception: {e}")
        return False

def test_redo_with_no_history():
    """Test redo with no redo history"""
    try:
        editor = EditorClass()
        editor.insert(0, "Test")
        editor.redo()  # Should have no effect
        result = editor.get_text()
        if result == "Test":
            print("✓ test_redo_with_no_history: PASS")
            return True
        else:
            print(f"✗ test_redo_with_no_history: FAIL - Expected 'Test', got '{result}'")
            return False
    except Exception as e:
        print(f"✗ test_redo_with_no_history: FAIL - Exception: {e}")
        return False

def test_complex_sequence():
    """Test complex sequence of operations"""
    try:
        editor = EditorClass()
        editor.insert(0, "Hello")
        editor.insert(5, " World")
        editor.delete(5, 6)
        editor.undo()  # Undo delete
        result1 = editor.get_text()
        
        editor.undo()  # Undo insert " World"
        result2 = editor.get_text()
        
        editor.redo()  # Redo insert " World"
        result3 = editor.get_text()
        
        editor.redo()  # Redo delete
        result4 = editor.get_text()
        
        if result1 == "Hello World" and result2 == "Hello" and result3 == "Hello World" and result4 == "Hello":
            print("✓ test_complex_sequence: PASS")
            return True
        else:
            print(f"✗ test_complex_sequence: FAIL")
            print(f"  After undo delete: '{result1}' (expected 'Hello World')")
            print(f"  After undo insert: '{result2}' (expected 'Hello')")
            print(f"  After redo insert: '{result3}' (expected 'Hello World')")
            print(f"  After redo delete: '{result4}' (expected 'Hello')")
            return False
    except Exception as e:
        print(f"✗ test_complex_sequence: FAIL - Exception: {e}")
        return False

def test_delete_undo_redo():
    """Test delete with undo and redo"""
    try:
        editor = EditorClass()
        editor.insert(0, "ABCDEF")
        result1 = editor.get_text()
        
        editor.delete(2, 2)  # Delete "CD"
        result2 = editor.get_text()
        
        editor.undo()
        result3 = editor.get_text()
        
        editor.redo()
        result4 = editor.get_text()
        
        if result1 == "ABCDEF" and result2 == "ABEF" and result3 == "ABCDEF" and result4 == "ABEF":
            print("✓ test_delete_undo_redo: PASS")
            return True
        else:
            print(f"✗ test_delete_undo_redo: FAIL")
            print(f"  Initial: '{result1}' (expected 'ABCDEF')")
            print(f"  After delete: '{result2}' (expected 'ABEF')")
            print(f"  After undo: '{result3}' (expected 'ABCDEF')")
            print(f"  After redo: '{result4}' (expected 'ABEF')")
            return False
    except Exception as e:
        print(f"✗ test_delete_undo_redo: FAIL - Exception: {e}")
        return False

def test_edge_cases():
    """Test edge cases with insertions at different positions"""
    try:
        editor = EditorClass()
        
        # Insert at beginning
        editor.insert(0, "Start")
        result1 = editor.get_text()
        
        # Insert at end
        editor.insert(5, "End")
        result2 = editor.get_text()
        
        # Insert in middle
        editor.insert(5, "Middle")
        result3 = editor.get_text()
        
        if result1 == "Start" and result2 == "StartEnd" and result3 == "StartMiddleEnd":
            print("✓ test_edge_cases: PASS")
            return True
        else:
            print(f"✗ test_edge_cases: FAIL")
            print(f"  After insert at 0: '{result1}' (expected 'Start')")
            print(f"  After insert at end: '{result2}' (expected 'StartEnd')")
            print(f"  After insert in middle: '{result3}' (expected 'StartMiddleEnd')")
            return False
    except Exception as e:
        print(f"✗ test_edge_cases: FAIL - Exception: {e}")
        return False

def test_large_document():
    """Test with larger document"""
    try:
        editor = EditorClass()
        
        # Create a moderately sized document
        text = "A" * 1000
        editor.insert(0, text)
        result1 = editor.get_text()
        
        editor.insert(500, "B" * 100)
        result2 = editor.get_text()
        
        editor.undo()
        result3 = editor.get_text()
        
        if len(result1) == 1000 and len(result2) == 1100 and len(result3) == 1000:
            print("✓ test_large_document: PASS")
            return True
        else:
            print(f"✗ test_large_document: FAIL - Lengths: {len(result1)}, {len(result2)}, {len(result3)}")
            return False
    except Exception as e:
        print(f"✗ test_large_document: FAIL - Exception: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Text Editor Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_insert,
        test_multiple_inserts,
        test_basic_delete,
        test_simple_undo,
        test_simple_redo,
        test_multiple_undo_redo,
        test_edit_after_undo_clears_redo,
        test_undo_on_empty,
        test_redo_with_no_history,
        test_complex_sequence,
        test_delete_undo_redo,
        test_edge_cases,
        test_large_document,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Now let me create the solution file:

```python
#!/usr/bin/env python3

class Piece:
    """Represents a piece in the piece table"""
    def __init__(self, buffer_type, start, length):
        self.buffer_type = buffer_type  # 'original' or 'add'
        self.start = start
        self.length = length

class Operation:
    """Represents an operation for undo/redo"""
    def __init__(self, op_type, position, data):
        self.op_type = op_type  # 'insert' or 'delete'
        self.position = position
        self.data = data  # text for insert, (start, length, deleted_text) for delete

class TextEditor:
    """Text editor with piece table and undo/redo support"""
    
    def __init__(self):
        self.original_buffer = ""
        self.add_buffer = ""
        self.pieces = []
        self.undo_stack = []
        self.redo_stack = []
    
    def insert(self, position, text):
        """Insert text at the specified position"""
        if not text:
            return
        
        # Add text to add buffer
        add_start = len(self.add_buffer)
        self.add_buffer += text
        
        # Find the piece and offset where we need to insert
        new_pieces = []
        current_pos = 0
        inserted = False
        
        for piece in self.pieces:
            piece_end = current_pos + piece.length
            
            if not inserted and position <= current_pos:
                # Insert before this piece
                new_pieces.append(Piece('add', add_start, len(text)))
                new_pieces.append(piece)
                inserted = True
            elif not inserted and position >= piece_end:
                # Insert after this piece
                new_pieces.append(piece)
                current_pos = piece_end
            elif not inserted and current_pos < position < piece_end:
                # Split this piece
                offset = position - current_pos
                
                # First part of split piece
                new_pieces.append(Piece(piece.buffer_type, piece.start, offset))
                
                # Inserted piece
                new_pieces.append(Piece('add', add_start, len(text)))
                
                # Second part of split piece
                new_pieces.append(Piece(piece.buffer_type, piece.start + offset, piece.length - offset))
                
                inserted = True
                current_pos = piece_end
            else:
                new_pieces.append(piece)
                current_pos = piece_end
        
        if not inserted:
            # Insert at the end
            new_pieces.append(Piece('add', add_start, len(text)))
        
        self.pieces = new_pieces
        
        # Record operation for undo
        self.undo_stack.append(Operation('insert', position, text))
        self.redo_stack.clear()
    
    def delete(self, start, length):
        """Delete length characters starting at start position"""
        if length <= 0:
            return
        
        # Get the text being deleted for undo
        deleted_text = self.get_text()[start:start + length]
        
        end = start + length
        new_pieces = []
        current_pos = 0
        
        for piece in self.pieces:
            piece_end = current_pos + piece.length
            
            if piece_end <= start or current_pos >= end:
                # Piece is completely outside deletion range
                new_pieces.append(piece)
            elif current_pos >= start and piece_end <= end:
                # Piece is completely inside deletion range - skip it
                pass
            elif current_pos < start and piece_end > end:
                # Deletion range is completely inside this piece - split into two
                offset_start = start - current_pos
                offset_end = end - current_pos
                
                # Part before deletion
                new_pieces.append(Piece(piece.buffer_type, piece.start, offset_start))
                
                # Part after deletion
                new_pieces.append(Piece(piece.buffer_type, piece.start + offset_end, 
                                       piece.length - offset_end))
            elif current_pos < start:
                # Deletion starts in the middle of this piece
                offset = start - current_pos
                new_pieces.append(Piece(piece.buffer_type, piece.start, offset))
            else:
                # Deletion ends in the middle of this piece
                offset = end - current_pos
                new_pieces.append(Piece(piece.buffer_type, piece.start + offset, 
                                       piece.length - offset))
            
            current_pos = piece_end
        
        self.pieces = new_pieces
        
        # Record operation for undo
        self.undo_stack.append(Operation('delete', start, (start, length, deleted_text)))
        self.redo_stack.clear()
    
    def get_text(self):
        """Return the current document content"""
        result = []
        for piece in self.pieces:
            if piece.buffer_type == 'original':
                result.append(self.original_buffer[piece.start:piece.start + piece.length])
            else:  # 'add'
                result.append(self.add_buffer[piece.start:piece.start + piece.length])
        return ''.join(result)
    
    def undo(self):
        """Undo the last operation"""
        if not self.undo_stack:
            return
        
        operation = self.undo_stack.pop()
        self.redo_stack.append(operation)
        
        if operation.op_type == 'insert':
            # Undo insert by deleting
            position = operation.position
            text = operation.data
            self._delete_without_recording(position, len(text))
        else:  # 'delete'
            # Undo delete by inserting
            start, length, deleted_text = operation.data
            self._insert_without_recording(start, deleted_text)
    
    def redo(self):
        """Redo the last undone operation"""
        if not self.redo_stack:
            return
        
        operation = self.redo_stack.pop()
        self.undo_stack.append(operation)
        
        if operation.op_type == 'insert':
            # Redo insert
            position = operation.position
            text = operation.data
            self._insert_without_recording(position, text)
        else:  # 'delete'
            # Redo delete
            start, length, deleted_text = operation.data
            self._delete_without_recording(start, length)
    
    def _insert_without_recording(self, position, text):
        """Insert text without recording to undo stack"""
        if not text:
            return
        
        add_start = len(self.add_buffer)
        self.add_buffer += text
        
        new_pieces = []
        current_pos = 0
        inserted = False
        
        for piece in self.pieces:
            piece_end = current_pos + piece.length
            
            if not inserted and position <= current_pos:
                new_pieces.append(Piece('add', add_start, len(text)))
                new_pieces.append(piece)
                inserted = True
            elif not inserted and position >= piece_end:
                new_pieces.append(piece)
                current_pos = piece_end
            elif not inserted and current_pos < position < piece_end:
                offset = position - current_pos
                new_pieces.append(Piece(piece.buffer_type, piece.start, offset))
                new_pieces.append(Piece('add', add_start, len(text)))
                new_pieces.append(Piece(piece.buffer_type, piece.start + offset, 
                                       piece.length - offset))
                inserted = True
                current_pos = piece_end
            else:
                new_pieces.append(piece)
                current_pos = piece_end
        
        if not inserted:
            new_pieces.append(Piece('add', add_start, len(text)))
        
        self.pieces = new_pieces
    
    def _delete_without_recording(self, start, length):
        """Delete without recording to undo stack"""
        if length <= 0:
            return
        
        end = start + length
        new_pieces = []
        current_pos = 0
        
        for piece in self.pieces:
            piece_end = current_pos + piece.length
            
            if piece_end <= start or current_pos >= end:
                new_pieces.append(piece)
            elif current_pos >= start and piece_end <= end:
                pass
            elif current_pos < start and piece_end > end:
                offset_start = start - current_pos
                offset_end = end - current_pos
                new_pieces.append(Piece(piece.buffer_type, piece.start, offset_start))
                new_pieces.append(Piece(piece.buffer_type, piece.start + offset_end, 
                                       piece.length - offset_end))
            elif current_pos < start:
                offset = start - current_pos
                new_pieces.append(Piece(piece.buffer_type, piece.start, offset))
            else:
                offset = end - current_pos
                new_pieces.append(Piece(piece.buffer_type, piece.start + offset, 
                                       piece.length - offset))
            
            current_pos = piece_end
        
        self.pieces = new_pieces


# Aliases for different naming conventions
Editor = TextEditor
PieceTableEditor = TextEditor
DocumentEditor = TextEditor