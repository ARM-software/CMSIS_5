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

option(BENCHMARK "Benchmarking compiled" OFF)
option(EXTBENCH "Benchmarking with external traces" OFF)

project(Testing)

# Needed to find the config modules


set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)




add_subdirectory(../Source bin_dsp)
add_subdirectory(${ROOT}/CMSIS/NN/Source bin_nn)

add_library(TestingLib STATIC)
add_library(FrameworkLib STATIC)

if (BENCHMARK)
set(TESTSRC 
  Source/Benchmarks/BasicMathsBenchmarksF32.cpp
  Source/Benchmarks/BasicMathsBenchmarksQ31.cpp
  Source/Benchmarks/BasicMathsBenchmarksQ15.cpp
  Source/Benchmarks/BasicMathsBenchmarksQ7.cpp
  Source/Benchmarks/ComplexMathsBenchmarksF32.cpp
  Source/Benchmarks/ComplexMathsBenchmarksQ31.cpp
  Source/Benchmarks/ComplexMathsBenchmarksQ15.cpp
  Source/Benchmarks/FIRF32.cpp
  Source/Benchmarks/FIRQ31.cpp
  Source/Benchmarks/FIRQ15.cpp
  Source/Benchmarks/MISCF32.cpp
  Source/Benchmarks/MISCQ31.cpp
  Source/Benchmarks/MISCQ15.cpp
  Source/Benchmarks/MISCQ7.cpp
  Source/Benchmarks/DECIMF32.cpp
  Source/Benchmarks/DECIMQ31.cpp
  Source/Benchmarks/DECIMQ15.cpp
  Source/Benchmarks/BIQUADF32.cpp
  Source/Benchmarks/BIQUADF64.cpp
  Source/Benchmarks/ControllerF32.cpp
  Source/Benchmarks/ControllerQ31.cpp
  Source/Benchmarks/ControllerQ15.cpp
  Source/Benchmarks/FastMathF32.cpp
  Source/Benchmarks/FastMathQ31.cpp
  Source/Benchmarks/FastMathQ15.cpp
  Source/Benchmarks/SupportF32.cpp
  Source/Benchmarks/SupportBarF32.cpp
  Source/Benchmarks/SupportQ31.cpp
  Source/Benchmarks/SupportQ15.cpp
  Source/Benchmarks/SupportQ7.cpp
  Source/Benchmarks/UnaryF32.cpp
  Source/Benchmarks/UnaryF64.cpp
  Source/Benchmarks/UnaryQ31.cpp
  Source/Benchmarks/UnaryQ15.cpp
  Source/Benchmarks/BinaryF32.cpp
  Source/Benchmarks/BinaryQ31.cpp
  Source/Benchmarks/BinaryQ15.cpp
  Source/Benchmarks/TransformF32.cpp
  Source/Benchmarks/TransformQ31.cpp
  Source/Benchmarks/TransformQ15.cpp
  Source/Benchmarks/FullyConnectedBench.cpp
  Source/Benchmarks/PoolingBench.cpp
  )
target_include_directories(TestingLib PRIVATE Include/Benchmarks)
else()

set(TESTSRC 
  Source/Tests/BasicTestsF32.cpp
  Source/Tests/SVMF32.cpp
  Source/Tests/BayesF32.cpp
  Source/Tests/StatsTestsF32.cpp
  Source/Tests/SupportTestsF32.cpp
  Source/Tests/SupportBarTestsF32.cpp
  Source/Tests/DistanceTestsF32.cpp
  Source/Tests/DistanceTestsU32.cpp
  Source/Tests/FullyConnected.cpp
  Source/Tests/Pooling.cpp
  Source/Tests/Softmax.cpp
  Source/Tests/NNSupport.cpp
  )
target_include_directories(TestingLib PRIVATE Include/Tests)

endif()

set(FRAMEWORKSRC
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
  FrameworkSource/Calibrate.cpp
  )
# Change behavior of configBoot for scatter file
# We use the linker files from older test framework because bigger sections are needed.
# We should migrate the linker files to this new framework.
set(TESTFRAMEWORK ON)
include(config)

# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(TestingLib)
disableOptimization(FrameworkLib)


target_sources(TestingLib PRIVATE ${TESTSRC})
target_sources(TestingLib PRIVATE testmain.cpp)
target_sources(TestingLib PRIVATE GeneratedSource/TestDesc.cpp)
  

target_sources(FrameworkLib PRIVATE ${FRAMEWORKSRC})

if (EXTBENCH)
  target_compile_definitions(FrameworkLib PUBLIC EXTBENCH)
endif()

### Includes
target_link_libraries(TestingLib PRIVATE CMSISDSP)
target_link_libraries(TestingLib PRIVATE CMSISNN)
target_include_directories(TestingLib PRIVATE FrameworkInclude)
target_include_directories(TestingLib PRIVATE GeneratedInclude)

configLib(TestingLib ${ROOT})

configLib(FrameworkLib ${ROOT})
target_include_directories(FrameworkLib PRIVATE FrameworkInclude)
# arm_math.h is needed for q7,q15,q31 types
# which are used for access to pattern files.
target_include_directories(FrameworkLib PRIVATE ${ROOT}/CMSIS/DSP/Include)


# Because we need access to core include for
# timing features in the test framework.
# So we need to identify the core
# then reference the right include folder
set_platform_core()
core_includes(FrameworkLib)

add_executable(Testing main.cpp)

# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(Testing)

configApp(Testing ${ROOT})

target_link_libraries(Testing PRIVATE TestingLib)
target_link_libraries(Testing PRIVATE FrameworkLib)

writeConfig(${CMAKE_CURRENT_BINARY_DIR}/currentConfig.csv)





