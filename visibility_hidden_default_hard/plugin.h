#ifndef PLUGIN_H
#define PLUGIN_H

#ifdef __cplusplus
extern "C" {
#endif

int plugin_load();
void plugin_execute(const char* command);

#ifdef __cplusplus
}
#endif

#endif