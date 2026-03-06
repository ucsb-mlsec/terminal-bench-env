cmake_minimum_required(VERSION 3.10)
project(MathUtils C)

# Add the static library
add_library(mathutils STATIC mathutils.c)

# Install the library
install(TARGETS mathutils
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

# Install the header file
install(FILES mathutils.h DESTINATION include)