cmake_minimum_required(VERSION 3.10)
project(SampleProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

set(SOURCE_FILES
    src/main.cpp
    src/utils.cpp
    src/parser.cpp
    src/handler.cpp
    src/processor.cpp
)

add_executable(SampleProject ${SOURCE_FILES})

target_compile_options(SampleProject PRIVATE
    -O2
    -march=native
)

install(TARGETS SampleProject DESTINATION bin)