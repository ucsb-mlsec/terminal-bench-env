cmake_minimum_required(VERSION 3.10)
project(calculator CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(CALCOPS REQUIRED calcops)

add_executable(calc_app main.cpp)

target_include_directories(calc_app PRIVATE ${CALCOPS_INCLUDE_DIRS})
target_link_directories(calc_app PRIVATE ${CALCOPS_LIBRARY_DIRS})
target_link_libraries(calc_app ${CALCOPS_LIBRARIES})