# CUDA Configuration CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

# Set CUDA target properties
add_library(cuda_utils STATIC cuda_utils.cu)
set_target_properties(cuda_utils
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Check CUDA version requirements
if(CMAKE_CUDA_COMPILER_VERSION LESS_THEN 10.0)
    message(FATAL_ERROR "CUDA 10.0 or higher is required")
endif()

# Configure CUDA compilation flags
list(APPEND 
    "-gencode=arch=compute_75,code=sm_75"
    "-gencode=arch=compute_80,code=sm_80"
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_NVCC_FLAGS}")

# Set CUDA architectures for different compute capabilities
set(CMAKE_CUDA_ARCHITECTURES "compute.75;86;89")

if(ENABLE_TENSOR_CORES)
    list(APPEND CMAKE_CUDA_FLAGS "-use_fast_math")
    list(APPEND CMAKE_CUDA_FLAGS "-DENABLE_TENSOR_CORES")
    message(STATUS "Tensor Core support enabled")

# Find CUDA libraries
find_library(CUDA_CUBLAS_LIB
    NAMES cublas
    PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
)

target_link_libraries(cuda_utils PUBLIC ${CUDA_cublas_LIBRARY)

find_library(CUDA_CUSPARSE_LIB
    NAMES cusparse
    PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
)

set(CUDA_LIBRARY_PATH "/usr/local/cuda/lib64")
set(CUDA_LIBRARY_PATH "/usr/local/cuda/lib64")

# Find cuDNN library if available
if(USE_CUDNN)
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        PATHS /usr/local/cuda/lib64
        DOC "Path to cuDNN library"
    )
    if(CUDNN_LIBRARY STREQUAL "CUDNN_LIBRARY-NOTFOUND" OR CUDNN_FOUND STREQUAL Off)
        message(WARNING "cuDNN library not found")
    endif()
endif()