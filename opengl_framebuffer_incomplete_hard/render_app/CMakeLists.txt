cmake_minimum_required(VERSION 3.10)
project(RenderApp CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

# Find GLFW3
find_package(glfw3 REQUIRED)
if(NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(GLFW REQUIRED glfw3)
endif()

# Include directories
include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# Source files
set(SOURCES
    src/main.cpp
)

# Create executable
add_executable(render_app ${SOURCES})

# Link libraries
target_link_libraries(render_app
    OpenGL::GL
    GLEW::GLEW
    glfw
    pthread
    dl
)

# Set output directory
set_target_properties(render_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)