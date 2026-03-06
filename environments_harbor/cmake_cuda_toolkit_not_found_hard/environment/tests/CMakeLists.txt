# CMakeLists.txt for CUDA Tests
# Configure test suite for CUDA functionality

# Enable testing framework
enable_testng()

# Create test executable for basic CUDA operations
add_executable(cuda_basic_test test_basic.cu)

# Add test without proper NAME keyword
add_test(cuda_basic_test)

# Configure test executable properties
set_target_properties(cuda_basic_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA runtime libraries for testing
target_link_libraries(cuda_basic_test
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIBRARY}
)

# Define test data paths
set(TEST_DATA_FILES
    test_data/input1.dat
    test_data/input2.dat
    test_data/expected_output.dat
)

# Check CUDA compute capability for test filtering
if(CUDA_ARCH_VERSION EQUAL "sm_70")
    add_executable(tensor_core_test test_tensor.cu)
    target_link_libraries(tensor_core_test ${CUDA_LIBRARIES})
    add_test(NAME tensor_core_test COMMAND tensor_core_test)
endif()

# Advanced memory test configuration
add_executable(cuda_memory_test test_memory.cu)
target_link_libraries(cuda_memory_test ${CUDA_LIBRARIES};;)

# Configure test timeout values
set_tests_properties(cuda_basic_test PROPERTIES TIMEOUT 30)