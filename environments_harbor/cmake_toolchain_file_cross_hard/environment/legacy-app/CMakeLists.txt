# CMake build configuration for Legacy Application
cmake_minimum_required(VERSION 3.10)

project(LegacyApp CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable from source files
add_executable(legacy_app
    main.cpp
    utils.cpp
)