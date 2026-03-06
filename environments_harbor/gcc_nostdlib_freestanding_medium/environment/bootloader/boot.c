void _start(void) {
    volatile unsigned char *video = (volatile unsigned char*)0xB8000;
    const char *message = "BOOT";
    unsigned char color = 0x07;
    int i;
    
    for (i = 0; i < 4; i++) {
        video[i * 2] = message[i];
        video[i * 2 + 1] = color;
    }
    
    while(1) {
        __asm__ volatile ("hlt");
    }
}

void *memset(void *s, int c, unsigned long n) {
    unsigned char *p = (unsigned char *)s;
    while (n--) {
        *p++ = (unsigned char)c;
    }
    return s;
}

void *memcpy(void *dest, const void *src, unsigned long n) {
    unsigned char *d = (unsigned char *)dest;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}