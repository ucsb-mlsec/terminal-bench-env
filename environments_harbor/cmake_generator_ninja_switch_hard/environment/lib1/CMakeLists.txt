# lib1/CMakeLists.txt
add_library(lib1 STATIC lib1.cpp)

target_include_directories(lib1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(lib1 PROPERTIES
    OUTPUT_NAME "lib1"
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)