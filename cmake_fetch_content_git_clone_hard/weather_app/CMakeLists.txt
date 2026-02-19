cmake_minimum_required(VERSION 3.14)
project(weather_app)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)

FetchContent_MakeAvailable(nlohmann_json)

add_executable(weather_processor src/main.cpp)

target_link_libraries(weather_processor PRIVATE nlohmann_json::nlohmann_json)