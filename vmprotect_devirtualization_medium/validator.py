#!/usr/bin/env python3

import sys
import base64
from functools import reduce

# Obfuscated dead code functions
def _0x1a2b3c(_l1l1l1):
    """Dead code: never called"""
    _0O0O0O = [chr(i) for i in range(65, 91)]
    return ''.join([_0O0O0O[i % 26] for i in range(_l1l1l1)])

def _x9z8y7(_a1b2c3):
    """Dead code: decoy function"""
    _decode = base64.b64decode(b'VmFsaWRhdGlvbg==').decode()
    return len(_decode) * _a1b2c3 if _a1b2c3 > 0 else 0

# Obfuscated helper functions
def _l1Il1I(_s):
    """Check length constraint"""
    _0x5f5f = lambda x: len(x) == 16
    _l1l1l = _0x5f5f(_s)
    if not _l1l1l:
        _dummy = _x9z8y7(5)  # Dead code call
    return _l1l1l

def _O0O0O0(_ch):
    """Check if uppercase letter"""
    _0xABCD = ord('A')
    _0xDCBA = ord('Z')
    _val = ord(_ch)
    return _0xABCD <= _val <= _0xDCBA

def _I1I1I1(_ch):
    """Check if digit"""
    _x1x1 = ord('0')
    _x9x9 = ord('9')
    _v = ord(_ch)
    return _x1x1 <= _v <= _x9x9

def _zXzXzX(_key_str):
    """Obfuscated position validation"""
    _result = True
    _l = len(_key_str)
    
    # Complex opaque predicate (always true)
    if (lambda: sum([1 for _ in range(10)]) == 10)():
        for _idx in range(_l):
            _char = _key_str[_idx]
            # Even positions must be letters
            if _idx % 2 == 0:
                if not _O0O0O0(_char):
                    _result = False
                    break
            # Odd positions must be digits
            else:
                if not _I1I1I1(_char):
                    _result = False
                    break
    
    # Dead code branch
    if not _result and False:
        _dummy = _0x1a2b3c(10)
    
    return _result

def _qWeRtY(_k):
    """Calculate digit sum with obfuscation"""
    _digits = []
    _temp = list(_k)
    
    # Opaque predicate loop
    for _i in range(len(_temp) if True or _x9z8y7(1) else 0):
        if _I1I1I1(_temp[_i]):
            _digits.append(int(_temp[_i]))
    
    # Complex sum calculation
    _sum = reduce(lambda a, b: a + b, _digits, 0)
    
    # Dead code
    if _sum < 0:
        _sum = _x9z8y7(_sum)
    
    return _sum

def _aSdFgH(_license):
    """Validate digit sum equals 36"""
    _target = 36
    _calculated = _qWeRtY(_license)
    
    # Obfuscated comparison
    _is_valid = (_calculated == _target)
    
    # Opaque predicate
    if (lambda x: x * 2 == x + x)(5):
        return _is_valid
    
    return False

def _ZxCvBn(_key):
    """Check first character constraint"""
    _valid_starts = ['A', 'B', 'C']
    _first = _key[0] if len(_key) > 0 else ''
    
    # Obfuscated membership test
    _check = False
    for _v in _valid_starts:
        if _first == _v:
            _check = True
            break
    
    # Dead code
    if not _check and (lambda: False)():
        _check = _x9z8y7(0) > 0
    
    return _check

def _MnBvCx(_k):
    """Check last character constraint"""
    _ends = [ord('0'), ord('5'), ord('9')]
    _last_char = _k[-1] if len(_k) > 0 else ''
    _last_ord = ord(_last_char) if _last_char else -1
    
    _valid = False
    
    # Complex validation
    for _e in _ends:
        if _last_ord == _e:
            _valid = True
    
    # Opaque predicate
    if (lambda: 1 + 1 == 2)():
        return _valid
    
    return False

def _PoIuYt(_license_key):
    """Check alphanumeric constraint"""
    _valid = True
    
    for _c in _license_key:
        _is_upper = _O0O0O0(_c)
        _is_digit = _I1I1I1(_c)
        
        if not (_is_upper or _is_digit):
            _valid = False
            break
    
    # Dead code branch
    if _valid and False:
        _dummy = _0x1a2b3c(20)
    
    return _valid

# Main obfuscated validator
def _VALIDATOR(_license_key_input):
    """Main validation function with heavy obfuscation"""
    
    # Dead code initialization
    _x0x0x0 = base64.b64encode(b'license').decode()
    _y0y0y0 = [i for i in range(100) if i % 2 == 0]
    
    # Validation checks with obfuscated control flow
    _checks = []
    
    # Check 1: Length
    _c1 = _l1Il1I(_license_key_input)
    _checks.append(_c1)
    
    # Opaque predicate
    if (lambda: True and True)():
        # Check 2: Position validation
        _c2 = _zXzXzX(_license_key_input)
        _checks.append(_c2)
    
    # Check 3: Alphanumeric
    _c3 = _PoIuYt(_license_key_input)
    _checks.append(_c3)
    
    # Dead code
    if len(_checks) < 0:
        _checks = [False]
    
    # Check 4: Digit sum
    _c4 = _aSdFgH(_license_key_input)
    _checks.append(_c4)
    
    # Check 5: First character
    _c5 = _ZxCvBn(_license_key_input)
    _checks.append(_c5)
    
    # Check 6: Last character
    _c6 = _MnBvCx(_license_key_input)
    _checks.append(_c6)
    
    # Obfuscated all() check
    _final_result = True
    for _chk in _checks:
        if not _chk:
            _final_result = False
            break
    
    # More dead code
    if _final_result and False:
        _final_result = (lambda: False)()
    
    # Complex return with opaque predicate
    if (lambda x: x == x)(True):
        return _final_result
    
    return False

# Dead code class
class _ObfuscatedClass:
    def __init__(self):
        self._data = _0x1a2b3c(15)
    
    def _method1(self):
        return base64.b64encode(b'dummy').decode()
    
    def _method2(self, x):
        return x * 2 if x > 0 else 0

# More dead code
def _unused_func_1():
    _temp = [i ** 2 for i in range(10)]
    return sum(_temp)

def _unused_func_2(_x, _y):
    return _x + _y if _x > _y else _y - _x

# Entry point
if __name__ == '__main__':
    # Dead code initialization
    _obj = _ObfuscatedClass()
    _unused = _unused_func_1()
    
    # Get license key from command line
    if len(sys.argv) < 2:
        # No key provided
        sys.exit(1)
    
    _input_key = sys.argv[1]
    
    # Obfuscated validation call
    _is_valid_license = False
    
    # Opaque predicate
    if (lambda: 5 > 3)():
        _is_valid_license = _VALIDATOR(_input_key)
    
    # Dead code branch
    if _is_valid_license and (lambda: False)():
        _is_valid_license = False
    
    # Exit with appropriate code
    if _is_valid_license:
        sys.exit(0)
    else:
        sys.exit(1)