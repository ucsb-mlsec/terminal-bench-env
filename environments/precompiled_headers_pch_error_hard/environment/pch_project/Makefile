CXX = g++
CXXFLAGS = -std=c++11 -Wall
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
TARGET = calculator

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

# Precompiled header - WRONG PATH HERE (bug)
PCH_SRC = $(SRC_DIR)/common.h
PCH_OUT = $(SRC_DIR)/common.h.gch

.PHONY: all clean

all: $(TARGET)

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Rule to generate precompiled header - uses wrong source path
$(PCH_OUT): $(PCH_SRC)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -x c++-header $< -o $@

# Compilation rule with incorrect PCH include path (bug)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(PCH_OUT) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -include $(INC_DIR)/common.h -c $< -o $@

# Linking rule
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@

# Clean rule
clean:
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET)
	rm -f $(PCH_OUT)
	rm -f $(INC_DIR)/common.h.gch

# Individual object file dependencies (ensuring PCH is built first)
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(PCH_OUT)
$(OBJ_DIR)/add.o: $(SRC_DIR)/add.cpp $(PCH_OUT)
$(OBJ_DIR)/multiply.o: $(SRC_DIR)/multiply.cpp $(PCH_OUT)