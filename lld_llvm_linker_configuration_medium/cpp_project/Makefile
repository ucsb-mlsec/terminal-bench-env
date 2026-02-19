CXX = g++
CXXFLAGS = -std=c++11 -Wall
TARGET = app
OBJS = main.o math_ops.o string_utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

math_ops.o: math_ops.cpp
	$(CXX) $(CXXFLAGS) -c math_ops.cpp

string_utils.o: string_utils.cpp
	$(CXX) $(CXXFLAGS) -c string_utils.cpp

clean:
	rm -f *.o $(TARGET)