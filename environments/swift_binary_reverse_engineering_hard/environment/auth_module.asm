.section __TEXT,__text,regular,pure_instructions
.build_version ios, 14, 0
.globl _swift_auth_performAPIAuthentication
.p2align 2

; -- Begin function swift_auth_performAPIAuthentication
_swift_auth_performAPIAuthentication:
; %bb.0:
    stp x29, x30, [sp, #-96]!
    stp x28, x27, [sp, #16]
    stp x26, x25, [sp, #32]
    stp x24, x23, [sp, #48]
    stp x22, x21, [sp, #64]
    stp x20, x19, [sp, #80]
    mov x29, sp
    sub sp, sp, #64
    
    ; Load AES encryption key reference
    adrp x19, .LCPI0_0@PAGE
    add x19, x19, .LCPI0_0@PAGEOFF
    
    ; Load API token string reference
    adrp x20, .L__unnamed_1@PAGE
    add x20, x20, .L__unnamed_1@PAGEOFF
    
    ; Store parameters
    str x0, [sp, #32]
    str x1, [sp, #40]
    
    ; Swift runtime retain call
    mov x0, x19
    bl _swift_retain
    
    ; Allocate buffer for encryption context
    mov w0, #256
    bl _swift_slowAlloc
    mov x21, x0
    
    ; Initialize encryption context with key
    mov x0, x21
    mov x1, x19
    mov w2, #32
    bl _memcpy
    
    ; Load token string length
    mov x0, x20
    bl _strlen
    mov x22, x0
    
    ; Allocate buffer for authorization header
    add w0, w22, #128
    bl _swift_slowAlloc
    mov x23, x0
    
.LBB0_1:
    ; Copy token to header buffer
    mov x0, x23
    mov x1, x20
    mov x2, x22
    bl _memcpy
    
    ; Setup for encryption operation
    ldr x0, [sp, #32]
    cbz x0, .LBB0_3
    
.LBB0_2:
    ; Perform encryption setup
    mov x0, x21
    mov x1, x23
    mov x2, x22
    adrp x3, .LCPI0_1@PAGE
    add x3, x3, .LCPI0_1@PAGEOFF
    bl _swift_crypto_aes_init
    
    ; Load key bytes into registers for processing
    ldr x8, [x19]
    ldr x9, [x19, #8]
    ldr x10, [x19, #16]
    ldr x11, [x19, #24]
    
    ; Store key components to stack
    str x8, [sp, #48]
    str x9, [sp, #56]
    stp x10, x11, [sp]
    
    ; Call encryption function
    mov x0, x21
    ldr x1, [sp, #32]
    ldr x2, [sp, #40]
    bl _swift_crypto_encrypt_buffer
    
    mov x24, x0
    cbnz x0, .LBB0_4
    b .LBB0_5
    
.LBB0_3:
    ; Error path - null context
    mov w0, #0
    b .LBB0_6
    
.LBB0_4:
    ; Setup authorization header
    mov x0, x23
    mov x1, x24
    mov x2, x22
    bl _swift_auth_setAuthHeader
    
    ; Create API request object
    adrp x8, _swift_auth_request_vtable@GOTPAGE
    ldr x8, [x8, _swift_auth_request_vtable@GOTPAGEOFF]
    mov x0, x8
    bl _swift_allocObject
    mov x25, x0
    
    ; Set request properties
    str x23, [x25, #16]
    str x24, [x25, #24]
    
    ; Set encryption key reference
    str x21, [x25, #32]
    
    ; Call request dispatch
    mov x0, x25
    ldr x8, [x25]
    ldr x8, [x8, #80]
    blr x8
    
    mov x26, x0
    b .LBB0_6
    
.LBB0_5:
    ; Error handling path
    mov x0, x21
    bl _swift_slowDealloc
    
    mov x0, x23
    bl _swift_slowDealloc
    
    mov w0, #0
    b .LBB0_6
    
.LBB0_6:
    ; Cleanup and return
    mov x19, x0
    
    ; Release token string
    mov x0, x20
    bl _swift_release
    
    ; Return result
    mov x0, x19
    
    add sp, sp, #64
    ldp x20, x19, [sp, #80]
    ldp x22, x21, [sp, #64]
    ldp x24, x23, [sp, #48]
    ldp x26, x25, [sp, #32]
    ldp x28, x27, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

; -- End function

.section __TEXT,__const
.p2align 3

.LCPI0_0:
    .byte 0x2f, 0x8e, 0x41, 0x6a, 0xb5, 0x93, 0x27, 0xd4
    .byte 0x5c, 0x71, 0xe9, 0x3b, 0xa8, 0x64, 0xf2, 0x1d
    .byte 0x9a, 0x47, 0xc3, 0x85, 0x6f, 0xb2, 0x38, 0xd9
    .byte 0x7e, 0x51, 0xa6, 0x2c, 0xf8, 0x94, 0x6b, 0x3e

.LCPI0_1:
    .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

.section __TEXT,__cstring,cstring_literals

.L__unnamed_1:
    .asciz "Bearer_a7f3k9m2x8q5w1p6v4j0z"

.L__unnamed_2:
    .asciz "Authorization"

.L__unnamed_3:
    .asciz "Content-Type"

.L__unnamed_4:
    .asciz "application/json"

.section __DATA,__const
.p2align 3

_swift_auth_request_vtable:
    .quad 0
    .quad 0
    .quad _swift_auth_request_destroy
    .quad _swift_auth_request_execute

.subsections_via_symbols