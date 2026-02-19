# CMake Toolchain File for Clang Compiler
# This file forces CMake to use Clang/Clang++ instead of the default compiler

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Set compiler identification to ensure CMake recognizes these as valid compilers
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_CXX_COMPILER_ID "Clang")

# Force compiler check to use the specified compilers
set(CMAKE_C_COMPILER_FORCED TRUE)
set(CMAKE_CXX_COMPILER_FORCED TRUE)

# Set the compiler architecture (if needed)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Ensure CMake doesn't try to detect the compiler again
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_CXX_COMPILER_WORKS TRUE)