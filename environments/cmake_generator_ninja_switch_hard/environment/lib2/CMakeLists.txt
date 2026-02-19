cmake_minimum_required(VERSION 3.10)

add_library(lib2 STATIC lib2.cpp)

target_include_directories(lib2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(TARGET lib2 PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Building lib2...")