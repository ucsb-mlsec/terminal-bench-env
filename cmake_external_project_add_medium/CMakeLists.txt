cmake_minimum_required(VERSION 3.10)
project(myapp CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main application executable
add_executable(myapp src/main.cpp)

# TODO: Add external library integration here
# The stringutils library in external/stringutils/ needs to be built
# and linked against this target

# Attempted include (doesn't work without building the library first)
# include_directories(external/stringutils/include)

# This won't work without proper external project setup
# target_link_libraries(myapp stringutils)

# Installation rules
install(TARGETS myapp DESTINATION bin)