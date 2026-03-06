cmake_minimum_required(VERSION 3.10)
project(LogLib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library source files
add_library(logger SHARED 
    src/logger.cpp
)

# Include directories for the library
target_include_directories(logger PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Command-line utility
add_executable(logcli 
    src/main.cpp
)

# Link the executable with the library
target_link_libraries(logcli PRIVATE logger)

# Install targets - BUGGY: Missing DESTINATION for library and executable
install(TARGETS logger
    LIBRARY
    ARCHIVE
    RUNTIME
)

install(TARGETS logcli
    RUNTIME
)

# Install header files
install(FILES 
    include/logger.h
    DESTINATION include
)

# Install documentation
install(FILES 
    README.md
    DESTINATION share/doc/loglib
)

# CPack configuration
set(CPACK_PACKAGE_NAME "LogLib")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VENDOR "LogLib Developers")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A simple C++ logging library")
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)