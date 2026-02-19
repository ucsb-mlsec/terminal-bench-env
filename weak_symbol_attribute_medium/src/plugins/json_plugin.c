#include <stdio.h>
#include <string.h>

void format_json(const char* message)
{
    printf("{\"log\": \"%s\", \"format\": \"json\"}\n", message);
}