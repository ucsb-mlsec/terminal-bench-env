#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

void utils_init();
const char* utils_format(const char* input);
void utils_cleanup();

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */