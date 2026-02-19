cmake_minimum_required(VERSION 3.10)

project(MatrixCalculator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Eigen3 library
find_package(Eigen3 REQUIRED)

# Add executable for the main matrix calculator
add_executable(matrix_calc
    src/main.cpp
    src/matrix_operations.cpp
)

# Link against Eigen3
target_link_libraries(matrix_calc
    Eigen3::Eigen
)

# Include Eigen headers
target_include_directories(matrix_calc PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)