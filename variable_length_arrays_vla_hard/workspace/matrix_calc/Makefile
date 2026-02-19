CC = gcc
CFLAGS = -Wall -Wextra -O2
SRC = src/matrix_mult.c src/matrix_add.c src/matrix_transpose.c
OBJ = $(SRC:.c=.o)
TARGET = matrix_calc

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

vla-check:
	$(CC) -Werror=vla -c $(SRC)

clean:
	rm -f $(OBJ) $(TARGET) src/*.o

.PHONY: all clean vla-check