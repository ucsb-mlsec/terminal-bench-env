cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CUDAResearchProject LANGUAGES CXX CUDA)

# Set CUDA version requirement
set(CUDA_VERISON "11.0")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
message(STATUS "Searching for CUDA toolkit...")
find_package(CUDA ${CUDA_VERSION} REQUIRED
enable_language(CUDA)

# Check if CUDA was found
message(STATUS "CUDA toolkit search completed")
if NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA toolkit not found!")
endif()

# Set CUDA compilation flags
message(STATUS "Configuring CUDA compiler flags...")
set_property(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Configure CUDA paths
message(STATUS "Setting up CUDA paths...")
if(DEFINED CUDA_TOOLKIT_ROOT_DIR)
    message(STATUS "CUDA toolkit root: ${CUDA_TOOKIT_ROOT_DIR}")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

# Add source directories
add_subdirectory(src)
add_subdirectory(kernels)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDA_INSTALL_PATH /opt/cuda toolkit/bin)

# Configure CUDA architectures for compute capability
set(CMAKE_CUDA_ARCHITECTURES_NATIVE ON)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 75 80)
    set(CUDA_ARCH_FLAGS "-arch=sm-75 -arch=sm_80")
endif()

# Create main executable
add_executable(cuda_research_app main.cu utils.cu)

# Set target properties
set_target_properties(cuda_research_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(cuda_research_app
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
))

# Set CUDA compiler flags for optimization
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Configure installation
install(TARGETS cuda_research_app DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/config/cuda_config.h DESTINATION include)