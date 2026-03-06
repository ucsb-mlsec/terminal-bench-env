BITS 32
ORG 0x1000000

; Security audit test program - complex register manipulation
; Initial state: EAX=0x42, EBX=0x10, ECX=0x5, EDX=0x0

; Phase 1: Basic arithmetic transformations
add eax, ebx          ; EAX = 0x42 + 0x10 = 0x52
mov edx, eax          ; EDX = 0x52
imul ebx, ecx         ; EBX = 0x10 * 0x5 = 0x50
xor eax, ebx          ; EAX = 0x52 XOR 0x50 = 0x02

; Phase 2: Bit manipulation and shifts
shl ecx, 4            ; ECX = 0x5 << 4 = 0x50
or edx, ecx           ; EDX = 0x52 OR 0x50 = 0x52
rol ebx, 8            ; EBX = rotate left 0x50 by 8 = 0x5000
add eax, 0x100        ; EAX = 0x02 + 0x100 = 0x102

; Phase 3: Complex interdependent operations
mov ecx, eax          ; ECX = 0x102
shl eax, 3            ; EAX = 0x102 << 3 = 0x810
sub ebx, edx          ; EBX = 0x5000 - 0x52 = 0x4FAE
xor edx, ebx          ; EDX = 0x52 XOR 0x4FAE = 0x4FFC

; Phase 4: More arithmetic mixing
add ecx, ebx          ; ECX = 0x102 + 0x4FAE = 0x50B0
and eax, 0xFFF        ; EAX = 0x810 AND 0xFFF = 0x810
inc edx               ; EDX = 0x4FFC + 1 = 0x4FFD
dec edx               ; EDX = 0x4FFD - 1 = 0x4FFC

; Phase 5: Final transformations
shr ecx, 2            ; ECX = 0x50B0 >> 2 = 0x142C
not eax               ; EAX = NOT 0x810 = 0xFFFFF7EF
and eax, edx          ; EAX = 0xFFFFF7EF AND 0x4FFC = 0x47EC
add ebx, ecx          ; EBX = 0x4FAE + 0x142C = 0x63DA
xor ecx, 0xAAAA       ; ECX = 0x142C XOR 0xAAAA = 0xBE86

; Termination
hlt