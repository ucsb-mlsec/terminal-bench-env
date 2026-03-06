#ifndef CONFIG_H
#define CONFIG_H

/* Configuration and utility functions */

/* Core configuration functions - used */
void init_system(void);
void load_configuration(const char* filename);
void save_configuration(const char* filename);
int get_config_value(const char* key);
void set_config_value(const char* key, int value);

/* Legacy configuration functions - potentially unused */
void reset_to_defaults(void);
void export_config_xml(const char* filename);
void import_config_xml(const char* filename);
void validate_config_schema(void);
void backup_configuration(void);
void restore_configuration(void);

/* Debugging and diagnostic functions - potentially unused */
void dump_config_to_console(void);
void print_config_statistics(void);
void verify_config_integrity(void);
int check_config_version(void);

/* Utility functions */
char* get_config_string(const char* key);
void free_config_string(char* str);

/* Configuration data */
extern const char* default_config_path;
extern int config_version;

#endif