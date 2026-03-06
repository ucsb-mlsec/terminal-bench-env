add_executable(myapp main.cpp)

target_link_libraries(myapp PRIVATE mylib)

target_include_directories(myapp PRIVATE 
    ${PROJECT_SOURCE_DIR}/src/mylib/include
)

# This RPATH configuration is problematic for multi-config generators
# It hardcodes the path without accounting for Debug/Release subdirectories
set_target_properties(myapp PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/src/mylib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)