CC = gcc
CFLAGS = -Wall -Werror -std=c99
TARGET = legacy_app
OBJS = main.o utils.o data_processor.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

utils.o: utils.c
	$(CC) $(CFLAGS) -c utils.c

data_processor.o: data_processor.c
	$(CC) $(CFLAGS) -c data_processor.c

clean:
	rm -f *.o $(TARGET)

test: $(TARGET)
	./$(TARGET)
	@if [ $$? -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "Tests failed!"; \
		exit 1; \
	fi

.PHONY: all clean test