# CMakeLists.txt for signal processor library and tool

# Define the shared library
add_library(signal SHARED signal_lib.cpp)

# Set library output directory - INTENTIONAL ERROR: malformed generator expression
set_target_properties(signal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/<CONFIG>
)

# Define the executable
add_executable(signal_tool main.cpp)

# Set executable output directory - INTENTIONAL ERROR: missing $ in generator expression
set_target_properties(signal_tool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/<CONFIG:>
)

# Link the executable against the library - INTENTIONAL ERROR: wrong bracket syntax
target_link_libraries(signal_tool PRIVATE signal)

# Additional library properties - INTENTIONAL ERROR: incorrect CONFIG reference
set_target_properties(signal PROPERTIES
    OUTPUT_NAME "signal"
    PREFIX "lib"
    SUFFIX ".so"
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/[CONFIG]
)

# Ensure proper visibility
target_compile_definitions(signal PRIVATE BUILDING_SIGNAL_LIB)

# Set C++ standard
set_property(TARGET signal PROPERTY CXX_STANDARD 11)
set_property(TARGET signal_tool PROPERTY CXX_STANDARD 11)