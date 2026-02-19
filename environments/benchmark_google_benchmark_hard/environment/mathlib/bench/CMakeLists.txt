add_executable(math_benchmarks math_bench.cpp)

target_include_directories(math_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(math_benchmarks
    mathlib
    benchmark
)