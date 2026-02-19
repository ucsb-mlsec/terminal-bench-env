cmake_minimum_required(VERSION 3.15)
project(SignalProcessor CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output directories for multi-config generators
# ERROR 1: Missing $ before generator expression
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/<CONFIG>)

# ERROR 2: Wrong syntax - using ${} around generator expression
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${<CONFIG>})

# Set archive output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add source subdirectory
add_subdirectory(src)

# ERROR 3: Incorrect generator expression in install command
install(TARGETS signal_lib signal_tool
    RUNTIME DESTINATION bin/<CONFIG>
    LIBRARY DESTINATION lib/<CONFIG>)