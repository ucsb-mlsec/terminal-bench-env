FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install build tools
RUN apt-get update && \
    apt-get install -y build-essential gcc make libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app

# Create the source files
RUN cat > /workspace/app/utils.h << 'EOF'
#ifndef UTILS_H
#define UTILS_H

int add_numbers(int a, int b);
int multiply_numbers(int a, int b);

#endif
EOF

RUN cat > /workspace/app/utils.c << 'EOF'
#include "utils.h"

int add_numbers(int a, int b) {
    return a + b;
}

int multiply_numbers(int a, int b) {
    return a * b;
}
EOF

RUN cat > /workspace/app/main.c << 'EOF'
#include <stdio.h>
#include "utils.h"

int main() {
    int result1 = add_numbers(5, 3);
    int result2 = multiply_numbers(4, 7);
    printf("Addition: %d\n", result1);
    printf("Multiplication: %d\n", result2);
    return 0;
}
EOF

RUN cat > /workspace/app/Makefile << 'EOF'
CC = gcc
CFLAGS = -Wall -g

myapp: main.o utils.o
	$(CC) $(CFLAGS) -o myapp main.o utils.o

main.o: main.c utils.h
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o myapp
EOF

WORKDIR /workspace/app