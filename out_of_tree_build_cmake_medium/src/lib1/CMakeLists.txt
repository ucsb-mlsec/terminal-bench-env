add_library(lib1 STATIC lib1.cpp)

# BROKEN: Using absolute path with CMAKE_SOURCE_DIR which causes issues
# when other targets try to include headers from this library
target_include_directories(lib1 PRIVATE ${CMAKE_SOURCE_DIR}/src/lib1)

# The problem here is that:
# 1. Using PRIVATE instead of PUBLIC means dependent targets can't find headers
# 2. Even if scope was correct, using CMAKE_SOURCE_DIR makes assumptions about layout
# 3. Should use CMAKE_CURRENT_SOURCE_DIR for proper relative path handling
# 4. Missing proper interface includes for consumers of this library