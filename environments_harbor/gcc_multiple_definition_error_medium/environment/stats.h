#ifndef STATS_H
#define STATS_H

/* Global statistics counters */
extern int total_packets;
extern int total_bytes;
extern long long total_errors;

/* Function prototypes for statistics management */
void init_statistics(void);
void print_statistics(void);
void reset_statistics(void);
void increment_total_packets(void);

#endif /* STATS_H */