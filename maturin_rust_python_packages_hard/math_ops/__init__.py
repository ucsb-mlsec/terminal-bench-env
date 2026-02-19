"""High-performance mathematical operations using Rust"""

__version__ = '1.2.3'

try:
    from .math_ops import *
except ImportError as e:
    import sys
    print(f"Warning: Failed to import Rust extension module: {e}", file=sys.stderr)
    print("The package may not be built correctly.", file=sys.stderr)

__all__ = ['__version__']