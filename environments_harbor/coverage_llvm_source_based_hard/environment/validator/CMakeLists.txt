cmake_minimum_required(VERSION 3.10)
project(Validator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Coverage flags for clang/gcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate")

# Source files
add_library(validator_lib validator.cpp)

# Test executable
add_executable(test_validator test_validator.cpp)
target_link_libraries(test_validator validator_lib)