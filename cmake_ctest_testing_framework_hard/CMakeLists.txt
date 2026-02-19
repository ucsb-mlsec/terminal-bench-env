cmake_minimum_required(VERSION 3.10)
project(datalib)

# Find Python3 interpreter
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Enable testing support
enable_testing()

# Test configuration section
# TODO: Register pytest tests here
# The previous developer was supposed to integrate pytest with CTest
# but never completed the implementation

# Placeholder for test discovery
# find_program(PYTEST_EXECUTABLE pytest)

# Test registration would go here but is missing
# This is where add_test() commands should be added to register
# the pytest test suite with CTest

message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Project configured successfully")