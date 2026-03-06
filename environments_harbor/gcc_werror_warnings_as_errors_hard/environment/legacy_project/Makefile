CC=gcc
CFLAGS=-Wall -Wextra -Iinclude
LDFLAGS=
TARGET=bin/app
BUILDDIR=build
SRCDIR=src
INCDIR=include

SOURCES=$(SRCDIR)/main.c $(SRCDIR)/utils.c $(SRCDIR)/parser.c $(SRCDIR)/network.c $(SRCDIR)/database.c
OBJECTS=$(BUILDDIR)/main.o $(BUILDDIR)/utils.o $(BUILDDIR)/parser.o $(BUILDDIR)/network.o $(BUILDDIR)/database.o
HEADERS=$(INCDIR)/utils.h $(INCDIR)/parser.h $(INCDIR)/network.h $(INCDIR)/database.h

all: $(TARGET)

$(TARGET): $(OBJECTS) | bin
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

bin:
	mkdir -p bin

clean:
	rm -rf $(BUILDDIR) bin

.PHONY: all clean