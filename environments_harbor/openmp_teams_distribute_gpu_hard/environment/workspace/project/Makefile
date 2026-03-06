CC=gcc
CXX=g++
CFLAGS=-O3 -fopenmp -Wall
CXXFLAGS=-O3 -fopenmp -std=c++11 -Wall
LDFLAGS=-fopenmp

SRC_DIR=src
OBJ_DIR=obj
EXEC=compute_app

SOURCES_C=$(wildcard $(SRC_DIR)/*.c)
SOURCES_CPP=$(wildcard $(SRC_DIR)/*.cpp)
OBJECTS_C=$(SOURCES_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJECTS_CPP=$(SOURCES_CPP:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJECTS=$(OBJECTS_C) $(OBJECTS_CPP)

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(EXEC)

.PHONY: all clean