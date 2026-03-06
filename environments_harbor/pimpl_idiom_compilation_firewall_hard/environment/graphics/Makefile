CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Werror -Iinclude
SRC = src/Canvas.cpp src/Palette.cpp src/Pipeline.cpp
OBJ = $(SRC:.cpp=.o)
TEST_SRC = test/main.cpp
TEST_OBJ = test/main.o
TARGET = test/main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) $(TEST_OBJ) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TARGET)