cmake_minimum_required(VERSION 3.12)
project(ternary_computing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # If not found, download it
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Core library sources
set(CORE_SOURCES
    src/core/trit.cpp
    src/core/tryte.cpp
    src/core/memory.cpp
    src/core/processor.cpp
)

# Create Python module
pybind11_add_module(_ternary_core
    src/bindings/bindings.cpp
    ${CORE_SOURCES}
)

# Include directories
target_include_directories(_ternary_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

# Set output directory
set_target_properties(_ternary_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/ternary
)

# Optional: Build standalone C++ library
option(BUILD_CPP_LIB "Build standalone C++ library" OFF)
if(BUILD_CPP_LIB)
    add_library(ternary_core STATIC ${CORE_SOURCES})
    target_include_directories(ternary_core PUBLIC src/core)
endif()

# Optional: Build C++ tests
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_ternary
        tests/test_trit.cpp
        ${CORE_SOURCES}
    )

    target_include_directories(test_ternary PRIVATE src/core)
    target_link_libraries(test_ternary GTest::GTest GTest::Main)

    add_test(NAME TernaryTests COMMAND test_ternary)
endif()
