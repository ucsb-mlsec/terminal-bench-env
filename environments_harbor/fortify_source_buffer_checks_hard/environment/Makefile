CC = gcc
CFLAGS = -O0 -D_FORTIFY_SOURCE=2 -U_FORTIFY_SOURCE -Wall -Wno-unused-variable -Wno-deprecated-declarations
LDFLAGS =
TARGET = textutil
SRC = textutil.c
OBJ = textutil.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $(SRC) -o $(OBJ)

clean:
	rm -f $(TARGET) $(OBJ)