cmake_minimum_required (VERSION 3.6)
cmake_policy(SET CMP0077 NEW)
# The tests are assuming that MATRIX_CHECK is enabled when building
# CMSIS-DSP.
set(MATRIXCHECK ON)
set(FASTMATHCOMPUTATIONS OFF)
option(DUMPPATTERN "Dump test patterns when test is failing" ON)
project(DSP_Lib_TestSuite)

# Needed to find the config modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)


set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)


file(GLOB MAIN "Common/src/*.c")
file(GLOB BASICMATH_TESTS "Common/src/basic_math_tests/*.c")
file(GLOB COMPLEXMATH_TESTS "Common/src/complex_math_tests/*.c")
file(GLOB CONTROLLER_TESTS "Common/src/controller_tests/*.c")
file(GLOB FASTMATH_TESTS "Common/src/fast_math_tests/*.c")
file(GLOB FILTERING_TESTS "Common/src/filtering_tests/*.c")
file(GLOB INTRINSINCS_TESTS "Common/src/intrinsics_tests/*.c")
file(GLOB MATRIX_TESTS "Common/src/matrix_tests/*.c")
file(GLOB STATISTICS_TESTS "Common/src/statistics_tests/*.c")
file(GLOB SUPPORT_TESTS "Common/src/support_tests/*.c")
file(GLOB TRANSFORM_TESTS "Common/src/transform_tests/*.c")
file(GLOB JTEST_MAIN "Common/JTest/src/*.c")

set(TESTSRC ${MAIN}
  ${BASICMATH_TESTS}
  ${COMPLEXMATH_TESTS}
  ${CONTROLLER_TESTS}
  ${FASTMATH_TESTS}
  ${FILTERING_TESTS}
  ${INTRINSINCS_TESTS}
  ${MATRIX_TESTS}
  ${STATISTICS_TESTS}
  ${SUPPORT_TESTS}
  ${TRANSFORM_TESTS}
  ${JTEST_MAIN}
  )

set(JINCS 
  Common/JTest/inc
  Common/JTest/inc/arr_desc
  Common/inc/basic_math_tests
  Common/inc/complex_math_tests
  Common/inc/controller_tests
  Common/inc/fast_math_tests
  Common/inc/filtering_tests
  Common/inc/intrinsics_tests
  Common/inc/matrix_tests
  Common/inc/statistics_tests 
  Common/inc/support_tests 
  Common/inc/transform_tests
  )

add_subdirectory(../Source bin_dsp)
add_subdirectory(RefLibs bin_ref)


add_executable(DSP_Lib_TestSuite)

if (DUMPPATTERN)
      target_compile_definitions(DSP_Lib_TestSuite PRIVATE DUMPPATTERN)
endif()

# Change behavior of configBoot for scatter file
set(TESTFRAMEWORK ON)

include(configBoot)

file(COPY ${ROOT}/CMSIS/DSP/Examples/ARM/boot/RTE_Components.h DESTINATION tempLink)

target_link_libraries(DSP_Lib_TestSuite PRIVATE CMSISDSP)
target_link_libraries(DSP_Lib_TestSuite PRIVATE DspRefLibs)

target_sources(DSP_Lib_TestSuite PRIVATE ${TESTSRC})

### Includes
target_include_directories(DSP_Lib_TestSuite PRIVATE "Common/inc")
target_include_directories(DSP_Lib_TestSuite PRIVATE "Common/inc/templates")
target_include_directories(DSP_Lib_TestSuite PRIVATE ${JINCS})


