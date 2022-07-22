cmake_minimum_required(VERSION 3.10)

project(FuzzyAdaptiveBinarization)

find_package(OpenCV REQUIRED)

add_library(
    FabLib
    lib/include/compute_utils.hpp
    lib/src/compute_utils.cpp
    lib/include/fuzzy_sat.hpp
    lib/src/fuzzy_sat.cpp
)

set_target_properties(
    FabLib
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        LINKER_LANGUAGE CXX
        SOVERSION 1
        VERSION 1.0.0
)

target_include_directories(
    FabLib
    PUBLIC
        lib/include/
)

target_link_libraries(
    FabLib
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        FabLib
        ${OpenCV_LIBS}
)
