CXX = g++
CXXFLAGS = -Wall -std=c++11

all: test_program

libmath.a: math_lib.o
	ar rcs libmath.a math_lib.o

math_lib.o: math_lib.cpp
	$(CXX) $(CXXFLAGS) -c math_lib.cpp

test_program test_main.o libmath.a
	$(CXX) $(CXXFLAGS) -o test_program test_main.o libmath.a

test_main.o: test_main.cpp
        $(CXX) $(CXXFLAGS) -c test_main.cpp

clean:
	rm -f *.o libmath.a test_program