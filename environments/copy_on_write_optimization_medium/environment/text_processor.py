#!/usr/bin/env python3

class TextBuffer:
    """
    A simple text buffer implementation that creates full copies on snapshot.
    This is intentionally inefficient to demonstrate the performance problem.
    """
    
    def __init__(self, initial_text=''):
        """Initialize buffer with optional text."""
        self._content = list(initial_text)
    
    def get_text(self):
        """Returns the full text content as a string."""
        return ''.join(self._content)
    
    def insert(self, position, text):
        """Inserts text at the specified position."""
        if position < 0 or position > len(self._content):
            raise IndexError(f"Position {position} out of range")
        for i, char in enumerate(text):
            self._content.insert(position + i, char)
    
    def delete(self, start, end):
        """Deletes characters from start to end position."""
        if start < 0 or end > len(self._content) or start > end:
            raise IndexError(f"Invalid range: {start} to {end}")
        del self._content[start:end]
    
    def replace(self, start, end, text):
        """Replaces characters from start to end with new text."""
        self.delete(start, end)
        self.insert(start, text)
    
    def snapshot(self):
        """
        Creates and returns a FULL COPY of the current buffer.
        This is intentionally inefficient - copies entire buffer content.
        """
        new_buffer = TextBuffer()
        # Deep copy the entire content list
        new_buffer._content = self._content.copy()
        return new_buffer
    
    def length(self):
        """Returns the current length of the buffer."""
        return len(self._content)
    
    def __repr__(self):
        """String representation for debugging."""
        preview = self.get_text()[:50]
        if len(self._content) > 50:
            preview += '...'
        return f"TextBuffer(length={self.length()}, preview='{preview}')"


if __name__ == '__main__':
    # Example usage
    buffer = TextBuffer('Hello World')
    print(f"Initial: {buffer.get_text()}")
    
    snap1 = buffer.snapshot()
    buffer.insert(5, ' Beautiful')
    print(f"After insert: {buffer.get_text()}")
    print(f"Snapshot 1: {snap1.get_text()}")
    
    buffer.replace(6, 15, 'Wonderful')
    print(f"After replace: {buffer.get_text()}")
    
    snap2 = buffer.snapshot()
    buffer.delete(0, 6)
    print(f"After delete: {buffer.get_text()}")
    print(f"Snapshot 2: {snap2.get_text()}")