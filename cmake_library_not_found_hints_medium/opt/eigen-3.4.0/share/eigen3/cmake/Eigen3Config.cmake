# Eigen3Config.cmake
# CMake package configuration file for Eigen3

# Compute paths
get_filename_component(EIGEN3_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(EIGEN3_INSTALL_PREFIX "${EIGEN3_CMAKE_DIR}/../../../" ABSOLUTE)

# Set version
set(EIGEN3_VERSION "3.4.0")
set(Eigen3_VERSION "3.4.0")

# Set include directory
set(EIGEN3_INCLUDE_DIR "${EIGEN3_INSTALL_PREFIX}/include/eigen3")
set(Eigen3_INCLUDE_DIR "${EIGEN3_INSTALL_PREFIX}/include/eigen3")

# Set found flag
set(EIGEN3_FOUND TRUE)
set(Eigen3_FOUND TRUE)

# Check if target already exists
if(NOT TARGET Eigen3::Eigen)
  # Create imported target
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  
  # Set include directories for the target
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()

# Provide legacy variables for compatibility
set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
set(EIGEN3_LIBRARIES "")
set(EIGEN3_DEFINITIONS "")

# Package found message
if(NOT Eigen3_FIND_QUIETLY)
  message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR} (found version \"${EIGEN3_VERSION}\")")
endif()