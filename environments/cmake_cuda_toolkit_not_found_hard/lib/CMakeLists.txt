# CMakeLists.txt for CUDA library component
# This file configures the CUDA-enabled library for the project

cmake_minimum_required(VERSION 3.18)

# Add library target with CUDA sources
# BUG: Missing opening parenthesis
add_library cuda_lib STATIC
    cuda_kernels.cu
    cuda_utils.cu
    device_functions.cu
)

# Set CUDA standard version
# BUG: Typo in CMAKE_CUDA_STANDARD (STANDART instead of STANDARD)
set(CMAKE_CUDA_STANDART 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure include directories for the library
# BUG: Typo in scope keyword (PUBLC instead of PUBLIC)
target_include_directories(cuda_lib
    PUBLC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA runtime library
# BUG: Undefined variable CUDA_CUDART_LIBRARY referenced but never found/set
target_link_libraries(cuda_lib
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
        ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES}
)

# Set CUDA properties for the target
# BUG: Wrong function name (set_properties instead of set_target_properties)
set_properties(TARGET cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Add CUDA architecture flags
# BUG: Missing closing quote mark causing unterminated string
set(CUDA_ARCH_FLAGS "-gencode arch=compute_70,code=sm_70)

# Set additional CUDA flags
set_target_properties(cuda_lib PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"
)