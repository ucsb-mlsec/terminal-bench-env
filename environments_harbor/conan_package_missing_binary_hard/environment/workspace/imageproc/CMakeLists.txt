cmake_minimum_required(VERSION 3.15)
project(imageproc CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(processor 
    src/processor.cpp
)

target_link_libraries(processor 
    ${CONAN_LIBS}
)

install(TARGETS processor DESTINATION bin)