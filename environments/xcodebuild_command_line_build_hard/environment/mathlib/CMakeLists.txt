cmake_minimum_required(VERSION 3.10)
project(MathLib)

# Missing C++ standard setting
# Missing RPATH configuration for runtime library loading

# Attempt to add subdirectories but no proper setup
add_subdirectory(src)
add_subdirectory(tests)

# No installation rules
# No proper library output directory configuration
# Missing CMAKE_INSTALL_RPATH settings