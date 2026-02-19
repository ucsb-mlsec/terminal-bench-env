#!/usr/bin/env python3

import stringtools
result = stringtools.reverse_string("hello")
assert result == "olleh", f"Expected 'olleh', got '{result}'"
print("Test passed!")