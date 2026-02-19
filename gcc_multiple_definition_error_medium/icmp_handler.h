#ifndef ICMP_HANDLER_H
#define ICMP_HANDLER_H

/* ICMP Protocol Handler Header */

/* Function prototypes */
void init_icmp_handler(void);
void process_icmp_packet(const char* data, int len);
int get_icmp_packet_count(void);

/* Global variables */
extern int icmp_packet_count;
extern int icmp_error_count;

#endif /* ICMP_HANDLER_H */