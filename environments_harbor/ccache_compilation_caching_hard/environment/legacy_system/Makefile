CXX = g++
CXXFLAGS = -std=c++14 -Iinclude -O2
TARGET = legacy_system
SRCDIR = src
OBJDIR = .

SOURCES = $(SRCDIR)/main.cpp \
          $(SRCDIR)/compute_engine.cpp \
          $(SRCDIR)/data_processor.cpp \
          $(SRCDIR)/io_handler.cpp \
          $(SRCDIR)/validator.cpp

OBJECTS = main.o \
          compute_engine.o \
          data_processor.o \
          io_handler.o \
          validator.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

main.o: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/main.cpp

compute_engine.o: $(SRCDIR)/compute_engine.cpp
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/compute_engine.cpp

data_processor.o: $(SRCDIR)/data_processor.cpp
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/data_processor.cpp

io_handler.o: $(SRCDIR)/io_handler.cpp
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/io_handler.cpp

validator.o: $(SRCDIR)/validator.cpp
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/validator.cpp

clean:
	rm -f $(OBJECTS) $(TARGET)