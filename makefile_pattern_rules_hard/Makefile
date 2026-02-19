CXX = g++
CXXFLAGS = -g -O0 -DDEBUG
INCLUDES = -Iinclude

SRCS = src/main.cpp src/utils.cpp src/parser.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = bin/app

all: $(TARGET)

bin:
	mkdir -p bin

src/main.o: src/main.cpp include/utils.hpp include/parser.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/main.cpp -o src/main.o

src/utils.o: src/utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/utils.cpp -o src/utils.o

src/parser.o: src/parser.cpp include/parser.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/parser.cpp -o src/parser.o

$(TARGET): bin $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f src/*.o
	rm -f $(TARGET)
	rm -rf bin

.PHONY: all clean