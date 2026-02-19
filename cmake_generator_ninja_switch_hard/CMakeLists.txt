cmake_minimum_required(VERSION 3.10)
project(MyApplication CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add library subdirectories
add_subdirectory(lib1)
add_subdirectory(lib2)

# Add application subdirectory
add_subdirectory(app)

# Custom target that may cause issues with Ninja
add_custom_target(build_all ALL)
add_dependencies(build_all lib1 lib2)