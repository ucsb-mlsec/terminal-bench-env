#!/usr/bin/env python3

def match(pattern: str, text: str) -> bool:
    """
    Match a pattern against text with support for:
    - Literal characters
    - '.' matches any single character
    - '*' means zero or more of the preceding element
    - '+' means one or more of the preceding element
    - '?' means zero or one of the preceding element
    
    This implementation has bugs in backtracking logic.
    """
    
    def match_helper(p_idx, t_idx):
        # Base case: end of pattern
        if p_idx >= len(pattern):
            return t_idx >= len(text)
        
        # Base case: end of text
        if t_idx >= len(text):
            # Check if remaining pattern can match empty string
            while p_idx < len(pattern):
                if p_idx + 1 < len(pattern) and pattern[p_idx + 1] in '*?':
                    p_idx += 2
                else:
                    return False
            return True
        
        # Look ahead for quantifiers
        if p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '*':
            # Zero or more: buggy - doesn't try all possibilities
            char = pattern[p_idx]
            # Bug: only tries greedy matching, no backtracking
            if char == '.' or char == text[t_idx]:
                # Try matching one more character (greedy)
                if match_helper(p_idx, t_idx + 1):
                    return True
            # Try skipping (zero matches)
            return match_helper(p_idx + 2, t_idx)
        
        elif p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '+':
            # One or more: buggy implementation
            char = pattern[p_idx]
            # Bug: doesn't properly handle backtracking
            if char == '.' or char == text[t_idx]:
                # Must match at least once
                t_idx += 1
                # Then treat as *
                while t_idx < len(text) and (char == '.' or char == text[t_idx]):
                    t_idx += 1
                # Bug: only tries maximum match, no backtracking
                return match_helper(p_idx + 2, t_idx)
            return False
        
        elif p_idx + 1 < len(pattern) and pattern[p_idx + 1] == '?':
            # Zero or one: buggy
            char = pattern[p_idx]
            # Bug: tries wrong order
            if char == '.' or char == text[t_idx]:
                # Try matching one
                return match_helper(p_idx + 2, t_idx + 1)
            # Try matching zero
            return match_helper(p_idx + 2, t_idx)
        
        else:
            # Single character match
            char = pattern[p_idx]
            if char == '.' or char == text[t_idx]:
                return match_helper(p_idx + 1, t_idx + 1)
            return False
    
    return match_helper(0, 0)