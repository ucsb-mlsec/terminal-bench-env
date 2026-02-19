#include <stdio.h>
#include <string.h>

void format_xml(const char* message)
{
    printf("<log format=\"xml\"><message>%s</message></log>\n", message);
}