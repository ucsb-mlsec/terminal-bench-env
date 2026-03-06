CXX = g++
CXXFLAGS = -std=c++11 -Wall -Iinclude
AR = ar
ARFLAGS = rcs

SRC_DIR = src
INCLUDE_DIR = include
TEST_DIR = tests
OBJ_DIR = obj

SOURCES = $(SRC_DIR)/vector_ops.cpp $(SRC_DIR)/matrix_ops.cpp $(SRC_DIR)/statistics.cpp
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

LIBRARY = libmathcore.a

TEST_SOURCES = $(TEST_DIR)/test_vector.cpp $(TEST_DIR)/test_matrix.cpp $(TEST_DIR)/test_stats.cpp
TEST_EXECUTABLES = $(TEST_DIR)/test_vector $(TEST_DIR)/test_matrix $(TEST_DIR)/test_stats

all: $(LIBRARY) $(TEST_EXECUTABLES)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBRARY): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

$(TEST_DIR)/test_vector: $(TEST_DIR)/test_vector.cpp $(LIBRARY)
	$(CXX) $(CXXFLAGS) $< -L. -lmathcore -o $@

$(TEST_DIR)/test_matrix: $(TEST_DIR)/test_matrix.cpp $(LIBRARY)
	$(CXX) $(CXXFLAGS) $< -L. -lmathcore -o $@

$(TEST_DIR)/test_stats: $(TEST_DIR)/test_stats.cpp $(LIBRARY)
	$(CXX) $(CXXFLAGS) $< -L. -lmathcore -o $@

clean:
	rm -rf $(OBJ_DIR)
	rm -f $(LIBRARY)
	rm -f $(TEST_EXECUTABLES)

.PHONY: all clean