#!/usr/bin/env python3

class RopeNode:
    def __init__(self, text=None, left=None, right=None):
        self.text = text
        self.left = left
        self.right = right
        
        if text is not None:
            # Bug: weight should be length of text for leaf nodes
            self.weight = len(text) + 1  # Off-by-one error
        elif left is not None:
            # Bug: weight should be total length of left subtree
            self.weight = self._calculate_weight(left) - 1  # Incorrect calculation
        else:
            self.weight = 0
    
    def _calculate_weight(self, node):
        if node is None:
            return 0
        if node.text is not None:
            return len(node.text)
        return self._calculate_weight(node.left) + self._calculate_weight(node.right)
    
    def is_leaf(self):
        return self.text is not None


class Rope:
    def __init__(self, text=""):
        if not text:
            self.root = None
        else:
            self.root = self._build_rope(text)
    
    def _build_rope(self, text):
        if len(text) <= 10:
            return RopeNode(text=text)
        
        mid = len(text) // 2
        left = self._build_rope(text[:mid])
        right = self._build_rope(text[mid:])
        return RopeNode(left=left, right=right)
    
    def length(self):
        # Bug: incorrect length calculation
        return self._length_helper(self.root) + 1  # Off-by-one error
    
    def _length_helper(self, node):
        if node is None:
            return 0
        if node.is_leaf():
            return len(node.text)
        return self._length_helper(node.left) + self._length_helper(node.right)
    
    def to_string(self):
        if self.root is None:
            return ""
        return self._to_string_helper(self.root)
    
    def _to_string_helper(self, node):
        if node is None:
            return ""
        if node.is_leaf():
            return node.text
        return self._to_string_helper(node.left) + self._to_string_helper(node.right)
    
    def insert(self, pos, text):
        if not text:
            return
        
        # Bug: incorrect position handling
        left_part = self.substring(0, pos - 1)  # Off-by-one error
        right_part = self.substring(pos, self.length())
        
        new_text = left_part + text + right_part
        self.root = self._build_rope(new_text)
    
    def delete(self, start, end):
        # Bug: incorrect range handling
        if start >= end:
            return
        
        left_part = self.substring(0, start)
        right_part = self.substring(end + 1, self.length())  # Off-by-one error
        
        new_text = left_part + right_part
        self.root = self._build_rope(new_text)
    
    def substring(self, start, end):
        if self.root is None:
            return ""
        
        # Bug: incorrect boundary checks
        if start < 0:
            start = 0
        if end > self.length():
            end = self.length()
        
        if start > end:  # Bug: should be >=
            return ""
        
        full_text = self.to_string()
        return full_text[start:end]
    
    def _substring_helper(self, node, start, end, offset):
        # Alternative buggy implementation (not used but shows another approach)
        if node is None:
            return ""
        
        if node.is_leaf():
            text_start = max(0, start - offset)
            text_end = min(len(node.text), end - offset)
            if text_start >= len(node.text) or text_end <= 0:
                return ""
            return node.text[text_start:text_end]
        
        left_length = self._length_helper(node.left)
        
        if end <= offset + left_length:
            return self._substring_helper(node.left, start, end, offset)
        elif start >= offset + left_length:
            return self._substring_helper(node.right, start, end, offset + left_length)
        else:
            left_part = self._substring_helper(node.left, start, end, offset)
            right_part = self._substring_helper(node.right, start, end, offset + left_length)
            return left_part + right_part


# Main execution for testing
if __name__ == "__main__":
    # Load test document
    try:
        with open('/workspace/test_document.txt', 'r') as f:
            document_text = f.read()
    except FileNotFoundError:
        document_text = "This is a test document for the rope data structure implementation. " * 10
    
    # Create rope with document
    rope = Rope(document_text)
    
    print(f"Initial length: {rope.length()}")
    
    # Perform test operations
    # 1. Insert " (CORRECTED)" at position 150
    rope.insert(150, " (CORRECTED)")
    
    # 2. Delete characters from position 50 to 75
    rope.delete(50, 75)
    
    # 3. Extract substring from position 100 to 200
    extracted = rope.substring(100, 200)
    
    # 4. Report final length
    final_length = rope.length()
    
    # Write results
    with open('/workspace/results.txt', 'w') as f:
        f.write(f"FINAL_LENGTH={final_length}\n")
        f.write(f"SUBSTRING={extracted}\n")
    
    print(f"Final length: {final_length}")
    print(f"Extracted substring: {extracted}")