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

target_compile_options(
    FabLib
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /fp:strict>
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -frounding-math -fsignaling-nans>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -ffp-model=strict>
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

# Tests
include(CTest)

add_executable(
    TestFab
    test/test_data.hpp
    test/test_data.cpp
    test/test_utils.hpp
    test/test_utils.cpp
    test/test_cf12_images.cpp
    test/test_cho_images.cpp
    test/test_ham_images.cpp
    test/test_sat_image.cpp
    test/test_sug_images.cpp
    test/main.cpp
)

set_target_properties(
    TestFab
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(
    FabLib
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /fp:strict>
         $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -frounding-math -fsignaling-nans>
         $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -ffp-model=strict>
)

target_include_directories(
    TestFab
    PRIVATE
        test
)

target_link_libraries(
    TestFab
    PRIVATE
        FabLib
)

add_test(NAME TestFab COMMAND TestFab)
