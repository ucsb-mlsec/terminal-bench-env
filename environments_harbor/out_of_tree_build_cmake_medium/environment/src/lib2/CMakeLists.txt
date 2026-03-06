add_library(lib2 STATIC lib2.cpp)

target_include_directories(lib2 PRIVATE ${CMAKE_SOURCE_DIR}/src/lib2)