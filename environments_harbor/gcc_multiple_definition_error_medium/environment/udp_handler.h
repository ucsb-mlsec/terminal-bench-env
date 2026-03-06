#ifndef UDP_HANDLER_H
#define UDP_HANDLER_H

/* UDP Protocol Handler Module */

/* Function prototypes */
void init_udp_handler(void);
void process_udp_packet(const char* data, int len);
int get_udp_packet_count(void);

/* Global variables */
extern int udp_packet_count;
extern int udp_error_count;

#endif /* UDP_HANDLER_H */