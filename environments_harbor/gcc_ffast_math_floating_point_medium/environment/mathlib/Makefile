CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -std=c++11
SRC = mathlib.cpp matrix.cpp vector.cpp computations.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = libmath.a

all: $(TARGET)

$(TARGET): $(OBJ)
	ar rcs $(TARGET) $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean