cmake_minimum_required(VERSION 3.10)
project(MainProject CXX)

set(CMAKE_CXX_STANDARD 11)

# BROKEN: Using hardcoded absolute paths for subdirectories
add_subdirectory(/home/user/project/src/lib1 lib1)
add_subdirectory(/home/user/project/src/lib2 lib2)

# BROKEN: Using absolute path for include directories
include_directories(/home/user/project/src/lib1/include)
include_directories(/home/user/project/src/lib2/include)

# BROKEN: Referencing source files with hardcoded paths
add_executable(main_app /home/user/project/src/main.cpp)

# BROKEN: Trying to link with incorrect target names and paths
target_link_libraries(main_app /home/user/project/build/liblib1.a)
target_link_libraries(main_app /home/user/project/build/liblib2.a)