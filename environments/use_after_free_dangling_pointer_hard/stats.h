#ifndef STATS_H
#define STATS_H

/* Forward declarations */
struct LogEntry;
struct LogStats;

typedef struct LogEntry LogEntry;
typedef struct LogStats LogStats;

/* Function declarations */
void update_stats(LogEntry* entry, LogStats* stats);
void reset_stats(LogStats* stats);

#endif /* STATS_H */