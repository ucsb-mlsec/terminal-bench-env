cmake_minimum_required(VERSION 3.10)
project(mathlib VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set install prefix
set(CMAKE_INSTALL_PREFIX /usr/local)

# Create test executable
add_executable(test_mathlib test_mathlib.cpp)

# Install header file
install(FILES mathlib.h DESTINATION include)

# Install documentation
install(FILES README.md DESTINATION share/doc/mathlib)

# CPack configuration - INCOMPLETE/BROKEN
# Missing critical variables and settings
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Math utilities library")
set(CPACK_PACKAGE_VENDOR "MathLib Team")

# Missing CPACK_PACKAGE_NAME
# Missing CPACK_PACKAGE_VERSION
# Missing CPACK_GENERATOR or set incorrectly
# Missing resource file specifications

include(CPack)