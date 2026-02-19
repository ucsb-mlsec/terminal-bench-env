cmake_minimum_required(VERSION 3.10)
project(PluginSystem)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(core)
add_subdirectory(plugin)
add_subdirectory(app)