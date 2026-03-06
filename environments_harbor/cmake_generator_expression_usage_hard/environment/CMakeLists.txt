cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/mylib)
add_subdirectory(src/app)

enable_testing()