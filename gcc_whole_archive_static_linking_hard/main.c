#include <stdio.h>
#include <stdlib.h>

/* External plugin status functions - these are provided by the plugin libraries */
extern int sensor_a_get_status(void);
extern int sensor_b_get_status(void);
extern int sensor_c_get_status(void);

int main(void)
{
    int status_a, status_b, status_c;
    int overall_status = 0;
    
    printf("Starting sensor application...\n");
    printf("Checking plugin initialization status...\n\n");
    
    /* Check sensor A initialization status */
    status_a = sensor_a_get_status();
    if (status_a == 0) {
        printf("Sensor A: initialized\n");
    } else {
        printf("Sensor A: FAILED to initialize (error code: %d)\n", status_a);
        overall_status = 1;
    }
    
    /* Check sensor B initialization status */
    status_b = sensor_b_get_status();
    if (status_b == 0) {
        printf("Sensor B: initialized\n");
    } else {
        printf("Sensor B: FAILED to initialize (error code: %d)\n", status_b);
        overall_status = 1;
    }
    
    /* Check sensor C initialization status */
    status_c = sensor_c_get_status();
    if (status_c == 0) {
        printf("Sensor C: initialized\n");
    } else {
        printf("Sensor C: FAILED to initialize (error code: %d)\n", status_c);
        overall_status = 1;
    }
    
    printf("\n");
    if (overall_status == 0) {
        printf("All sensors initialized successfully\n");
    } else {
        printf("ERROR: One or more sensors failed to initialize\n");
    }
    
    return overall_status;
}