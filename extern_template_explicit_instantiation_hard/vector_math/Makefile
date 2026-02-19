CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wextra
TARGETS=test1 test2 test3

all: $(TARGETS)

test1: main1.o
	$(CXX) -o test1 main1.o

test2: main2.o
	$(CXX) -o test2 main2.o

test3: main3.o
	$(CXX) -o test3 main3.o

%.o: %.cpp vector.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(TARGETS)