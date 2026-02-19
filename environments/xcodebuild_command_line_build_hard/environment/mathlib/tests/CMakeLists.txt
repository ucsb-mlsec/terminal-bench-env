cmake_minimum_required(VERSION 3.10)

add_executable(test_math test_main.cpp)

target_link_libraries(test_math matlib)

target_include_directories(test_math PRIVATE ../include)