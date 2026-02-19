cmake_minimum_required(VERSION 3.10)

project(cuda_pool_simulation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pool_sim memory_pool.cpp)

target_compile_options(pool_sim PRIVATE -Wall -g)

# Optional sanitizers (uncomment to enable)
# target_compile_options(pool_sim PRIVATE -fsanitize=address -fsanitize=undefined)
# target_link_options(pool_sim PRIVATE -fsanitize=address -fsanitize=undefined)