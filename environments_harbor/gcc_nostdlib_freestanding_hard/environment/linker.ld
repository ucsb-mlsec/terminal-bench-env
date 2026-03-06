OUTPUT_FORMAT(binary)
ENTRY(start)

SECTIONS
{
    . = 0x7C00;
    
    .text : ALIGN(1)
    {
        *(.text)
        *(.text*)
    }
    
    .rodata : ALIGN(1)
    {
        *(.rodata)
        *(.rodata*)
    }
    
    .data : ALIGN(1)
    {
        *(.data)
        *(.data*)
    }
    
    .bss : ALIGN(1)
    {
        *(.bss)
        *(.bss*)
        *(COMMON)
    }
    
    /DISCARD/ :
    {
        *(.comment)
        *(.comment.*)
        *(.note)
        *(.note.*)
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.gcc_except_table)
        *(.gcc_except_table.*)
        *(.interp)
        *(.dynamic)
        *(.dynsym)
        *(.dynstr)
        *(.hash)
        *(.gnu.hash)
        *(.gnu.version)
        *(.gnu.version_r)
        *(.rel.*)
        *(.rela.*)
    }
}