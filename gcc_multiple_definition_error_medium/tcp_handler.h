#ifndef TCP_HANDLER_H
#define TCP_HANDLER_H

/* TCP Protocol Handler Header */

/* Function prototypes */
void init_tcp_handler(void);
void process_tcp_packet(const char* data, int len);
int get_tcp_packet_count(void);

/* Global variables */
extern int tcp_packet_count;
extern int tcp_error_count;

#endif /* TCP_HANDLER_H */