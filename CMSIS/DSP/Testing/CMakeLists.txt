cmake_minimum_required (VERSION 3.6)
cmake_policy(SET CMP0077 NEW)
# The tests are assuming that MATRIX_CHECK is enabled when building
# CMSIS-DSP.

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

function(writeConfig path)
  set(output "")
  list(APPEND output "OPTIMIZED,HARDFP,FASTMATH,NEON,UNROLL,ROUNDING,PLATFORM,CORE,COMPILER,VERSION\n")

  if (OPTIMIZED)
    list(APPEND output "1")
  else()
    list(APPEND output "0")
  endif()

  if (HARDFP)
    list(APPEND output ",1")
  else()
    list(APPEND output ",0")
  endif()

  if (FASTMATHCOMPUTATIONS)
    list(APPEND output ",1")
  else()
    list(APPEND output ",0")
  endif()

  if (NEON OR NEONEXPERIMENTAL)
    list(APPEND output ",1")
  else()
    list(APPEND output ",0")
  endif()

  if (LOOPUNROLL)
    list(APPEND output ",1")
  else()
    list(APPEND output ",0")
  endif()

  if (ROUNDING)
    list(APPEND output ",1")
  else()
    list(APPEND output ",0")
  endif()

  list(APPEND output ",${PLATFORMID}")
  list(APPEND output ",${COREID},")
  if (ARMAC6)
    list(APPEND output "AC6")
  elseif(GCC)
    list(APPEND output "GCC")
  endif()
  compilerVersion()
  list(APPEND output ",${COMPILERVERSION}")

  file(WRITE ${path} ${output})
  
 
endfunction()

project(Testing)

# Needed to find the config modules


set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(TESTSRC testmain.cpp
  FrameworkSource/Test.cpp
  FrameworkSource/Semihosting.cpp
  FrameworkSource/IORunner.cpp
  FrameworkSource/ArrayMemory.cpp
  FrameworkSource/Pattern.cpp
  FrameworkSource/PatternMgr.cpp
  FrameworkSource/Error.cpp
  FrameworkSource/FPGA.cpp
  FrameworkSource/Timing.cpp
  FrameworkSource/Generators.cpp
  Source/BasicTestsF32.cpp
  Source/BasicMathsBenchmarksF32.cpp
  Source/BasicMathsBenchmarksQ31.cpp
  Source/FullyConnected.cpp
  Source/FullyConnectedBench.cpp
  GeneratedSource/TestDesc.cpp
  )


add_subdirectory(../Source bin_dsp)
add_subdirectory(${ROOT}/CMSIS/NN/Source bin_nn)

add_library(TestingLib STATIC)


# Change behavior of configBoot for scatter file
# We use the linker files from older test framework because bigger sections are needed.
# We should migrate the linker files to this new framework.
set(TESTFRAMEWORK ON)
include(config)

# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(TestingLib)


target_sources(TestingLib PRIVATE ${TESTSRC})


### Includes
target_link_libraries(TestingLib PRIVATE CMSISDSP)
target_link_libraries(TestingLib PRIVATE CMSISNN)
target_include_directories(TestingLib PRIVATE Include)
target_include_directories(TestingLib PRIVATE FrameworkInclude)
target_include_directories(TestingLib PRIVATE GeneratedInclude)
configLib(TestingLib ${ROOT})

# Because we need access to core include for
# timing features in the test framework.
# So we need to identify the core
# then reference the right include folder
set_platform_core()
core_includes(TestingLib)

add_executable(Testing main.cpp)

# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(Testing)

configApp(Testing ${ROOT})

target_link_libraries(Testing PRIVATE TestingLib)

writeConfig("currentConfig.csv")





