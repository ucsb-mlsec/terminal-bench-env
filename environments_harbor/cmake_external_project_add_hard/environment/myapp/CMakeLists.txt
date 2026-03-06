cmake_minimum_required(VERSION 3.10)
project(MyApp)

include(ExternalProject)

# Attempt to add external mathutils library - HAS ERRORS
ExternalProject_Add(
    mathutils_external
    PREFIX ${CMAKE_BINARY_DIR}/external
    # Missing SOURCE_DIR parameter!
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

# Wrong variable name for install directory
set(MATHUTILS_INSTALL ${CMAKE_BINARY_DIR}/wrong_path)
set(MATHUTILS_INCLUDE_DIR ${MATHUTILS_INSTALL}/include)
set(MATHUTILS_LIB_DIR ${MATHUTILS_INSTALL}/lib)

# Missing include_directories() call!
# include_directories(${MATHUTILS_INCLUDE_DIR})

# Add main executable
add_executable(myapp main.cpp)

# Wrong library name and missing dependency declaration
# Missing add_dependencies(myapp mathutils_external)
target_link_libraries(myapp ${MATHUTILS_LIB_DIR}/libwrongname.a)

# Incorrect link directories
link_directories(${CMAKE_BINARY_DIR}/nonexistent)