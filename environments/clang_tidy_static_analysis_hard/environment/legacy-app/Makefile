CXX = g++
CXXFLAGS = -Wall -Wextra -I./include
SRCDIR = src
OBJDIR = obj
INCDIR = include
TARGET = legacy-app

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(INCDIR)/database.h $(INCDIR)/utils.h $(INCDIR)/network.h $(INCDIR)/processor.h

$(OBJDIR)/database.o: $(SRCDIR)/database.cpp $(INCDIR)/database.h

$(OBJDIR)/utils.o: $(SRCDIR)/utils.cpp $(INCDIR)/utils.h

$(OBJDIR)/network.o: $(SRCDIR)/network.cpp $(INCDIR)/network.h

$(OBJDIR)/processor.o: $(SRCDIR)/processor.cpp $(INCDIR)/processor.h $(INCDIR)/utils.h