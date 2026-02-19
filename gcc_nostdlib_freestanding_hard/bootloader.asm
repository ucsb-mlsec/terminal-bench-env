BITS 16
ORG 0x7C00

start:
    ; Clear interrupts and set up segments
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    ; Print boot message
    mov si, boot_msg
    call print_string

    ; Hang the system
    cli
hang:
    hlt
    jmp hang

print_string:
    lodsb
    or al, al
    jz done
    mov ah, 0x0E
    mov bh, 0x00
    mov bl, 0x07
    int 0x10
    jmp print_string
done:
    ret

boot_msg db 'Bootloader loaded!', 13, 10, 0

times 510-($-$$) db 0
dw 0xAA55