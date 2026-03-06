CXX = g++
CXXFLAGS = -std=c++11 -pthread -O2
TARGET = counter

all: $(TARGET)

$(TARGET): counter.cpp
	$(CXX) $(CXXFLAGS) counter.cpp -o $(TARGET)

race: CXXFLAGS += -fsanitize=thread
race: clean $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean race