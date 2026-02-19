#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "processing.h"
#include "config.h"

#define VERSION "1.0.2"
#define MAX_INPUT_SIZE 256

/* Command line options structure */
typedef struct {
    int verbose;
    int self_test;
    int process_mode;
    char input_file[MAX_INPUT_SIZE];
    char output_file[MAX_INPUT_SIZE];
} app_options_t;

/* Initialize default options */
static void init_options(app_options_t *opts) {
    opts->verbose = 0;
    opts->self_test = 0;
    opts->process_mode = 0;
    opts->input_file[0] = '\0';
    opts->output_file[0] = '\0';
}

/* Print usage information */
static void print_usage(const char *prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -v, --verbose       Enable verbose output\n");
    printf("  -t, --self-test     Run self-test and exit\n");
    printf("  -m, --mode <n>      Set processing mode (0-3)\n");
    printf("  -i, --input <file>  Input file path\n");
    printf("  -o, --output <file> Output file path\n");
    printf("  -h, --help          Show this help message\n");
    printf("  --version           Show version information\n");
}

/* Parse command line arguments */
static int parse_arguments(int argc, char *argv[], app_options_t *opts) {
    int i;
    
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            opts->verbose = 1;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--self-test") == 0) {
            opts->self_test = 1;
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) {
            if (i + 1 < argc) {
                opts->process_mode = atoi(argv[++i]);
                if (opts->process_mode < 0 || opts->process_mode > 3) {
                    fprintf(stderr, "Error: Invalid mode value\n");
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: --mode requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) {
                strncpy(opts->input_file, argv[++i], MAX_INPUT_SIZE - 1);
                opts->input_file[MAX_INPUT_SIZE - 1] = '\0';
            } else {
                fprintf(stderr, "Error: --input requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                strncpy(opts->output_file, argv[++i], MAX_INPUT_SIZE - 1);
                opts->output_file[MAX_INPUT_SIZE - 1] = '\0';
            } else {
                fprintf(stderr, "Error: --output requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "--version") == 0) {
            printf("Legacy App Version %s\n", VERSION);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    
    return 0;
}

/* Run self-test sequence */
static int run_self_test(int verbose) {
    int result = 0;
    
    if (verbose) {
        printf("Running self-test sequence...\n");
    }
    
    /* Test utility functions */
    if (verbose) {
        printf("  Testing utility functions...\n");
    }
    if (test_utils() != 0) {
        fprintf(stderr, "Self-test failed: utility functions\n");
        result = -1;
    }
    
    /* Test processing functions */
    if (verbose) {
        printf("  Testing processing functions...\n");
    }
    if (test_processing() != 0) {
        fprintf(stderr, "Self-test failed: processing functions\n");
        result = -1;
    }
    
    /* Test configuration functions */
    if (verbose) {
        printf("  Testing configuration functions...\n");
    }
    if (test_config() != 0) {
        fprintf(stderr, "Self-test failed: configuration functions\n");
        result = -1;
    }
    
    if (result == 0) {
        printf("Self-test PASSED\n");
    } else {
        printf("Self-test FAILED\n");
    }
    
    return result;
}

/* Main application entry point */
int main(int argc, char *argv[]) {
    app_options_t options;
    int result;
    
    /* Initialize options */
    init_options(&options);
    
    /* Parse command line arguments */
    if (parse_arguments(argc, argv, &options) != 0) {
        return EXIT_FAILURE;
    }
    
    /* Run self-test if requested */
    if (options.self_test) {
        result = run_self_test(options.verbose);
        return (result == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    
    /* Initialize configuration */
    if (options.verbose) {
        printf("Initializing configuration...\n");
    }
    if (init_config() != 0) {
        fprintf(stderr, "Error: Failed to initialize configuration\n");
        return EXIT_FAILURE;
    }
    
    /* Initialize utility subsystem */
    if (options.verbose) {
        printf("Initializing utilities...\n");
    }
    init_utils();
    
    /* Perform processing based on mode */
    if (options.verbose) {
        printf("Processing mode: %d\n", options.process_mode);
    }
    
    result = process_data(options.process_mode, options.input_file, 
                         options.output_file, options.verbose);
    
    if (result != 0) {
        fprintf(stderr, "Error: Processing failed\n");
        return EXIT_FAILURE;
    }
    
    if (options.verbose) {
        printf("Application completed successfully\n");
    }
    
    return EXIT_SUCCESS;
}