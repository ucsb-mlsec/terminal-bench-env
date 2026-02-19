MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 256K
    RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 64K
}

ENTRY(_start)

SECTIONS
{
    .text :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
        *(.text)
        *(.text*)
        . = ALIGN(4);
        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    } > FLASH

    .text.fast :
    {
        . = ALIGN(4);
        *(.text.fast)
        *(.text.fast*)
        . = ALIGN(4);
    } > FLASH

    .text.slow :
    {
        . = ALIGN(4);
        *(.text.slow)
        *(.text.slow*)
        . = ALIGN(4);
    } > FLASH

    .text.util :
    {
        . = ALIGN(4);
        *(.text.util)
        *(.text.util*)
        . = ALIGN(4);
    } > FLASH

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT> FLASH

    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    . = ALIGN(4);
    _end = .;

    /DISCARD/ :
    {
        *(.ARM.exidx*)
        *(.comment)
    }
}