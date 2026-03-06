/*******************************************************************************
 * @file    startup.c
 * @brief   Startup code for ARM Cortex-M4 bare-metal application
 * @details This file implements the startup sequence for ARM Cortex-M4 processor.
 *          On reset, the Cortex-M4 loads the initial stack pointer from address
 *          0x00000000, then fetches the reset handler address from 0x00000004
 *          and begins execution.
 *
 * Boot Sequence:
 * 1. Hardware loads SP from vector table[0]
 * 2. Hardware loads PC from vector table[1] (Reset_Handler)
 * 3. Reset_Handler copies .data section from Flash to RAM
 * 4. Reset_Handler zeroes .bss section in RAM
 * 5. Reset_Handler calls main()
 ******************************************************************************/

#include <stdint.h>

/*******************************************************************************
 * External symbols defined by the linker script
 * These symbols mark important memory boundaries
 ******************************************************************************/

/* Stack pointer initial value (top of stack) */
extern uint32_t _estack;

/* Data section - initialized variables in RAM */
extern uint32_t _sdata;  /* Start of .data in RAM */
extern uint32_t _edata;  /* End of .data in RAM */
extern uint32_t _sidata; /* Start of .data in Flash (initialization values) */

/* BSS section - uninitialized variables in RAM */
extern uint32_t _sbss;   /* Start of .bss */
extern uint32_t _ebss;   /* End of .bss */

/* Application entry point */
extern int main(void);

/*******************************************************************************
 * Function Prototypes
 ******************************************************************************/
void Reset_Handler(void);
void Default_Handler(void);

/* ARM Cortex-M4 Core Exception Handlers */
void NMI_Handler(void)                __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void)          __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void)          __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void)           __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void)         __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void)                __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void)           __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void)             __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void)            __attribute__((weak, alias("Default_Handler")));

/*******************************************************************************
 * Vector Table
 * Must be placed at the beginning of Flash memory (address 0x08000000)
 * The .isr_vector section is mapped by the linker script
 ******************************************************************************/
__attribute__((section(".isr_vector")))
void (* const vector_table[])(void) = {
    (void (*)(void))((uint32_t)&_estack),  /* Initial Stack Pointer */
    Reset_Handler,                          /* Reset Handler */
    NMI_Handler,                            /* NMI Handler */
    HardFault_Handler,                      /* Hard Fault Handler */
    MemManage_Handler,                      /* MPU Fault Handler */
    BusFault_Handler,                       /* Bus Fault Handler */
    UsageFault_Handler,                     /* Usage Fault Handler */
    0,                                      /* Reserved */
    0,                                      /* Reserved */
    0,                                      /* Reserved */
    0,                                      /* Reserved */
    SVC_Handler,                            /* SVCall Handler */
    DebugMon_Handler,                       /* Debug Monitor Handler */
    0,                                      /* Reserved */
    PendSV_Handler,                         /* PendSV Handler */
    SysTick_Handler,                        /* SysTick Handler */
    /* External Interrupts would follow here */
};

/*******************************************************************************
 * @brief  Reset Handler - Called on system reset
 * @details This is the first C code that executes after reset. It performs:
 *          - Copy initialized data from Flash to RAM (.data section)
 *          - Zero out uninitialized data in RAM (.bss section)
 *          - Call main() to start application
 ******************************************************************************/
void Reset_Handler(void)
{
    uint32_t *src, *dst;

    /* Copy the .data section from Flash to RAM
     * The .data section contains initialized global and static variables.
     * Their initial values are stored in Flash and must be copied to RAM.
     */
    src = &_sidata;  /* Source: initialization values in Flash */
    dst = &_sdata;   /* Destination: .data section in RAM */
    
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    /* Zero initialize the .bss section
     * The .bss section contains uninitialized global and static variables.
     * C standard requires these to be zero-initialized before main() runs.
     */
    dst = &_sbss;
    
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    /* Call the application's entry point */
    main();

    /* If main() returns (it shouldn't in embedded systems), loop forever */
    while (1) {
        /* Infinite loop */
    }
}

/*******************************************************************************
 * @brief  Default Handler for unused interrupts
 * @details This handler is called for any interrupt that doesn't have its own
 *          handler defined. It simply loops forever, which will help catch
 *          unexpected interrupts during debugging.
 ******************************************************************************/
void Default_Handler(void)
{
    /* Trap unexpected interrupts */
    while (1) {
        /* Infinite loop - attach debugger to investigate */
    }
}