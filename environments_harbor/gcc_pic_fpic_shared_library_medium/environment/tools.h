#ifndef TOOLS_H
#define TOOLS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Calculate the sum of two integers */
int calculate_result(int a, int b);

/* Process and display data value */
void process_data(int value);

/* Format an integer as a string */
const char* format_output(int num);

#ifdef __cplusplus
}
#endif

#endif /* TOOLS_H */