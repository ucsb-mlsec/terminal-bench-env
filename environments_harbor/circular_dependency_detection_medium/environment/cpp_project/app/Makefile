# Dependencies: libcore libutils libnetwork

CC = g++
CFLAGS = -Wall -Wextra -O2 -std=c++11 -I../libcore/include -I../libutils/include -I../libnetwork/include
LDFLAGS = -L../libcore -L../libutils -L../libnetwork
LIBS = -lcore -lutils -lnetwork

TARGET = app
SOURCES = main.cpp application.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)