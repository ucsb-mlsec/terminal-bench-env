CC = gcc
CFLAGS = -O2 -Wall -Wextra -fstack-protector-strong -D_FORTIFY_SOURCE=2
AR = ar
ARFLAGS = rcs

# Object files
OBJS = network.o

# Target library
TARGET = libnetwork.a

# Build the library
$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $(TARGET) $(OBJS)

# Compile network.c to network.o
network.o: network.c
	$(CC) $(CFLAGS) -c network.c -o network.o

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean