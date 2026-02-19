# Makefile for libui
# Dependencies: libutils libcore

CC = g++
CFLAGS = -Wall -Wextra -O2 -fPIC -Iinclude
LDFLAGS = -L../libutils -L../libcore -lutils -lcore
AR = ar
ARFLAGS = rcs

TARGET_STATIC = libui.a
TARGET_SHARED = libui.so
SRCDIR = src
OBJDIR = obj
INCDIR = include

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: all clean static shared

all: static shared

static: $(TARGET_STATIC)

shared: $(TARGET_SHARED)

$(TARGET_STATIC): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(TARGET_SHARED): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET_STATIC) $(TARGET_SHARED)

install: all
	cp $(TARGET_STATIC) /usr/local/lib/
	cp $(TARGET_SHARED) /usr/local/lib/
	cp -r $(INCDIR)/* /usr/local/include/