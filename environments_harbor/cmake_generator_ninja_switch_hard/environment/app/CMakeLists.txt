add_executable(myapp main.cpp)

include_directories(${CMAKE_SOURCE_DIR}/lib1)
include_directories(${CMAKE_SOURCE_DIR}/lib2)

target_link_libraries(myapp lib1)
target_link_libraries(myapp lib2)

add_dependencies(myapp lib1 lib2)