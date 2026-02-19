CXX=g++ -std=c++11 -O2
CXXFLAGS=-std=c++11 -O2
OBJECTS=main.o file1.o file2.o file3.o

.PHONY: all clean

all: program

program: $(OBJECTS)
	$(CXX) $(OBJECTS) -o program

%.o: %.cpp container.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o program