# CMakeLists.txt for mathlib static library

add_library(mathlib STATIC 
    add.cpp 
    subtract.cpp 
    multiply.cpp 
    divide.cpp
)

target_include_directories(mathlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})