#ifndef CORE_H
#define CORE_H

#ifdef __cplusplus
extern "C" {
#endif

void core_init();
int core_process(const char* data);
void core_shutdown();

#ifdef __cplusplus
}
#endif

#endif // CORE_H