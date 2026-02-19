#!/usr/bin/env python3
import sys
import base64

_ = lambda x: ''.join(chr(ord(c) ^ 42) for c in x)
__ = lambda x: [ord(c) for c in x]
___ = lambda x: ''.join(chr(c) for c in x)
____ = lambda x, y: x == y
_____ = lambda x: len(x)
______ = getattr(sys, 'exit')
_______ = getattr(sys, 'argv')
________ = getattr(sys.stdout, 'write')

O0O0O0 = base64.b64decode(b'S04tMlg5UC00UU04LVZTT0o=').decode()
O0O0O1 = lambda x: _____(x) == 19
O0O0O2 = lambda x: x[2] == '-' and x[8] == '-' and x[14] == '-'
O0O0O3 = lambda x: all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in x)

def O0O0O4(x):
    if not O0O0O1(x):
        return False
    if not O0O0O2(x):
        return False
    if not O0O0O3(x):
        return False
    return True

def O0O0O5(x):
    O0O0O6 = x.replace('-', '')
    O0O0O7 = 0
    for O0O0O8 in O0O0O6:
        if O0O0O8.isdigit():
            O0O0O7 += int(O0O0O8)
        else:
            O0O0O7 += ord(O0O0O8) - ord('A') + 10
    return O0O0O7

def O0O0O9(x):
    O0O0OA = [x[i:i+1] for i in range(_____(x))]
    O0O0OB = 0
    for O0O0OC in range(_____(O0O0OA)):
        if O0O0OA[O0O0OC] != '-':
            O0O0OB ^= ord(O0O0OA[O0O0OC])
    return O0O0OB

O0O0OD = lambda x: O0O0O5(x) % 37 == 23
O0O0OE = lambda x: O0O0O9(x) == 69

def O0O0OF(x):
    O0O0OG = x.replace('-', '')
    if _____(O0O0OG) != 15:
        return False
    O0O0OH = ''
    for O0O0OI in range(0, _____(O0O0OG), 3):
        O0O0OJ = O0O0OG[O0O0OI:O0O0OI+3]
        O0O0OK = 0
        for O0O0OL in O0O0OJ:
            if O0O0OL.isdigit():
                O0O0OK += int(O0O0OL)
            else:
                O0O0OK += ord(O0O0OL) - ord('A') + 1
        O0O0OH += chr(O0O0OK % 26 + ord('A'))
    return O0O0OH == 'KQVTS'

def O0O0OM(x):
    if not O0O0O4(x):
        return False
    if not O0O0OD(x):
        return False
    if not O0O0OE(x):
        return False
    if not O0O0OF(x):
        return False
    return True

if _____(_______ ) < 2:
    ________(base64.b64decode(b'SU5WQUxJRA==').decode() + '\n')
    ______(1)

O0O0ON = _______[1]

if O0O0OM(O0O0ON):
    ________(base64.b64decode(b'QUNUSVZBRUQ=').decode() + '\n')
    ______(0)
else:
    ________(base64.b64decode(b'SU5WQUxJRA==').decode() + '\n')
    ______(1)