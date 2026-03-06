add_library(plugin plugin.cpp)

target_include_directories(plugin PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
)

target_link_libraries(plugin core)