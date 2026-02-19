void boot_main(void) __attribute__((noreturn));

void write_char(char c, int pos) {
    unsigned short *video = (unsigned short *)0xB8000;
    unsigned short attr = 0x0F00;
    video[pos] = attr | c;
}

void write_string(const char *str) {
    unsigned short *video = (unsigned short *)0xB8000;
    unsigned short attr = 0x0F00;
    int i = 0;
    
    while (str[i] != '\0') {
        video[i] = attr | str[i];
        i++;
    }
}

void clear_screen(void) {
    unsigned short *video = (unsigned short *)0xB8000;
    int i;
    
    for (i = 0; i < 80 * 25; i++) {
        video[i] = 0x0F20;
    }
}

void boot_main(void) {
    const char *message = "Boot OK";
    
    clear_screen();
    write_string(message);
    
    while (1) {
        __asm__ volatile ("hlt");
    }
}