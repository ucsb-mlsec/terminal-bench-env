#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VERSION "1.0"

void print_help(const char *progname) {
    printf("Rescue Utilities v%s\n", VERSION);
    printf("Usage: %s [command] [arguments]\n\n", progname);
    printf("Available commands:\n");
    printf("  ls      - List directory contents\n");
    printf("  cat     - Display file contents\n");
    printf("  echo    - Print arguments to stdout\n");
    printf("  help    - Display this help message\n");
    printf("  --help  - Display this help message\n\n");
    printf("This is a minimal rescue utility for initramfs environments.\n");
}

int cmd_ls(int argc, char *argv[]) {
    printf("ls: Simulated directory listing\n");
    printf("  bin/\n");
    printf("  etc/\n");
    printf("  dev/\n");
    printf("  proc/\n");
    return 0;
}

int cmd_cat(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "cat: missing file operand\n");
        return 1;
    }
    printf("cat: would display contents of %s\n", argv[1]);
    return 0;
}

int cmd_echo(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        printf("%s", argv[i]);
        if (i < argc - 1) printf(" ");
    }
    printf("\n");
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    const char *command = argv[1];

    if (strcmp(command, "--help") == 0 || strcmp(command, "help") == 0) {
        print_help(argv[0]);
        return 0;
    }

    if (strcmp(command, "ls") == 0) {
        return cmd_ls(argc - 1, argv + 1);
    }

    if (strcmp(command, "cat") == 0) {
        return cmd_cat(argc - 1, argv + 1);
    }

    if (strcmp(command, "echo") == 0) {
        return cmd_echo(argc - 1, argv + 1);
    }

    fprintf(stderr, "Unknown command: %s\n", command);
    fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
    return 1;
}