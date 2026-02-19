/*
 * proctime.c - A simple proc filesystem module for system uptime
 * 
 * Original Author: J. Developer <jdev@oldcompany.com>
 * Created: March 2013
 * Last Modified: June 2015
 * 
 * Description: Creates /proc/proctime entry to display system uptime
 * Tested on: Linux 2.6.32 kernel
 * 
 * This module uses the proc filesystem to expose system uptime information
 * to userspace applications. Simple and straightforward implementation.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/time.h>
#include <asm/uaccess.h>

#define PROC_ENTRY_NAME "proctime"
#define BUFFER_SIZE 256

/* Global variables */
static struct proc_dir_entry *proc_entry;
static char output_buffer[BUFFER_SIZE];

/*
 * Function: get_system_uptime
 * 
 * Calculates system uptime in seconds
 * Returns: uptime in seconds as unsigned long
 */
static unsigned long get_system_uptime(void)
{
    struct timespec uptime;
    unsigned long uptime_secs;
    
    /* Get uptime - this worked fine in 2.6 kernels */
    do_posix_clock_monotonic_gettime(&uptime);
    uptime_secs = uptime.tv_sec;
    
    return uptime_secs;
}

/*
 * Function: format_uptime
 * 
 * Formats uptime into days, hours, minutes, seconds
 * Parameters:
 *   - total_secs: total uptime in seconds
 *   - buffer: output buffer to write formatted string
 * Returns: length of formatted string
 */
static int format_uptime(unsigned long total_secs, char *buffer)
{
    unsigned long days, hours, minutes, seconds;
    int len;
    
    days = total_secs / 86400;
    hours = (total_secs % 86400) / 3600;
    minutes = (total_secs % 3600) / 60;
    seconds = total_secs % 60;
    
    len = sprintf(buffer, "System Uptime: %lu days, %lu hours, %lu minutes, %lu seconds\n",
                  days, hours, minutes, seconds);
    
    return len;
}

/*
 * procfile_read - Called when /proc/proctime is read
 * 
 * Old-style proc read function (kernel 2.6 era)
 * This signature worked perfectly back in the day!
 */
static int procfile_read(char *buffer, char **start, off_t offset,
                         int count, int *eof, void *data)
{
    int len;
    unsigned long uptime_secs;
    
    /* Check if we already sent data */
    if (offset > 0) {
        *eof = 1;
        return 0;
    }
    
    /* Get current uptime */
    uptime_secs = get_system_uptime();
    
    /* Format the uptime data */
    len = format_uptime(uptime_secs, output_buffer);
    
    /* Copy to user buffer */
    if (len > count) {
        len = count;
    }
    
    memcpy(buffer, output_buffer, len);
    
    *eof = 1;
    return len;
}

/*
 * procfile_write - Called when /proc/proctime is written to
 * 
 * We don't actually do anything with writes, just acknowledge them
 */
static int procfile_write(struct file *file, const char *buffer,
                          unsigned long count, void *data)
{
    /* Just return the count - we ignore writes */
    printk(KERN_INFO "proctime: Write operation not supported\n");
    return count;
}

/*
 * Module initialization function
 * 
 * Creates the /proc/proctime entry using the tried-and-true
 * create_proc_entry function from 2.6 kernel days
 */
static int proctime_init(void)
{
    printk(KERN_INFO "proctime: Initializing module\n");
    
    /* Create proc entry - this API was stable for years! */
    proc_entry = create_proc_entry(PROC_ENTRY_NAME, 0644, NULL);
    
    if (proc_entry == NULL) {
        printk(KERN_ERR "proctime: Failed to create /proc/%s\n", PROC_ENTRY_NAME);
        return -ENOMEM;
    }
    
    /* Set up the proc file operations */
    proc_entry->read_proc = procfile_read;
    proc_entry->write_proc = procfile_write;
    proc_entry->owner = THIS_MODULE;
    proc_entry->mode = S_IFREG | S_IRUGO | S_IWUSR;
    proc_entry->uid = 0;
    proc_entry->gid = 0;
    proc_entry->size = BUFFER_SIZE;
    
    printk(KERN_INFO "proctime: /proc/%s created successfully\n", PROC_ENTRY_NAME);
    
    return 0;
}

/*
 * Module cleanup function
 * 
 * Removes the /proc/proctime entry
 */
static void proctime_exit(void)
{
    printk(KERN_INFO "proctime: Cleaning up module\n");
    
    /* Remove proc entry - using old API */
    if (proc_entry) {
        remove_proc_entry(PROC_ENTRY_NAME, NULL);
        printk(KERN_INFO "proctime: /proc/%s removed\n", PROC_ENTRY_NAME);
    }
}

/* Register init and exit functions */
module_init(proctime_init);
module_exit(proctime_exit);