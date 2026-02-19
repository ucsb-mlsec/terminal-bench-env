/*
 * Module A Driver
 * 
 * A simple Linux kernel module for demonstration purposes
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Developer A");
MODULE_DESCRIPTION("Module A driver");
MODULE_VERSION("1.0");

static int __init module_a_init(void)
{
    printk(KERN_INFO "Module A: Initializing module\n");
    printk(KERN_INFO "Module A: Driver loaded successfully\n");
    printk(KERN_INFO "Module A: Version 1.0\n");
    
    return 0;
}

static void __exit module_a_exit(void)
{
    printk(KERN_INFO "Module A: Cleaning up module\n");
    printk(KERN_INFO "Module A: Driver unloaded successfully\n");
}

module_init(module_a_init);
module_exit(module_a_exit);