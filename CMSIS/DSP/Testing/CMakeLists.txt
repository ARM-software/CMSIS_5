cmake_minimum_required (VERSION 3.6)
cmake_policy(SET CMP0077 NEW)
include(CMakePrintHelpers)

# The tests are assuming that MATRIX_CHECK is enabled when building
# CMSIS-DSP.

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

function(writeConfig path)
  set(output "")
  list(APPEND output "OPTIMIZED,HARDFP,FASTMATH,NEON,HELIUM,UNROLL,ROUNDING,PLATFORM,CORE,COMPILER,VERSION\n")

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

  if (HELIUM OR MVEI OR MVEF)
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
option(NN "NN Tests included" OFF)
option(REFLIB "Use already built reference lib" OFF)

project(Testing)

# Needed to find the config modules


set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)


# Change behavior of configBoot for scatter file
# We use the linker files from older test framework because bigger sections are needed.
# We should migrate the linker files to this new framework.
set(TESTFRAMEWORK ON)
include(config)

if (NOT REFLIB)
add_subdirectory(../Source bin_dsp)
endif()

if (NN)
add_subdirectory(${ROOT}/CMSIS/NN/Source bin_nn)
endif()

add_library(TestingLib STATIC)
add_library(FrameworkLib STATIC)

if (BENCHMARK)
set (NNSRC
  Source/Benchmarks/FullyConnectedBench.cpp
  Source/Benchmarks/PoolingBench.cpp
  )

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
  )
target_include_directories(TestingLib PRIVATE Include/Benchmarks)
else()

set(NNSRC
  Source/Tests/NNSupport.cpp
  Source/Tests/Pooling.cpp
  Source/Tests/Softmax.cpp
  Source/Tests/FullyConnected.cpp
  )

set(TESTSRC 
  Source/Tests/BasicTestsF32.cpp
  Source/Tests/BasicTestsQ31.cpp
  Source/Tests/BasicTestsQ15.cpp
  Source/Tests/BasicTestsQ7.cpp
  Source/Tests/ComplexTestsF32.cpp
  Source/Tests/ComplexTestsQ31.cpp
  Source/Tests/ComplexTestsQ15.cpp
  Source/Tests/SVMF32.cpp
  Source/Tests/BayesF32.cpp
  Source/Tests/TransformCF64.cpp
  Source/Tests/TransformCF32.cpp
  Source/Tests/TransformRF64.cpp
  Source/Tests/TransformRF32.cpp
  Source/Tests/TransformCQ31.cpp
  Source/Tests/TransformRQ31.cpp
  Source/Tests/TransformCQ15.cpp
  Source/Tests/TransformRQ15.cpp
  Source/Tests/StatsTestsF32.cpp
  Source/Tests/StatsTestsF64.cpp
  Source/Tests/StatsTestsQ31.cpp
  Source/Tests/StatsTestsQ15.cpp
  Source/Tests/StatsTestsQ7.cpp
  Source/Tests/FastMathF32.cpp
  Source/Tests/FastMathQ31.cpp
  Source/Tests/FastMathQ15.cpp
  Source/Tests/SupportTestsF32.cpp
  Source/Tests/SupportTestsQ31.cpp
  Source/Tests/SupportTestsQ15.cpp
  Source/Tests/SupportTestsQ7.cpp
  Source/Tests/SupportBarTestsF32.cpp
  Source/Tests/DistanceTestsF32.cpp
  Source/Tests/DistanceTestsU32.cpp
  Source/Tests/UnaryTestsQ31.cpp
  Source/Tests/UnaryTestsQ15.cpp
  Source/Tests/UnaryTestsF32.cpp
  Source/Tests/UnaryTestsF64.cpp
  Source/Tests/BinaryTestsF32.cpp
  Source/Tests/BinaryTestsQ31.cpp
  Source/Tests/BinaryTestsQ15.cpp
  Source/Tests/DECIMF32.cpp
  Source/Tests/DECIMQ31.cpp
  Source/Tests/DECIMQ15.cpp
  Source/Tests/MISCF32.cpp
  Source/Tests/MISCQ31.cpp
  Source/Tests/MISCQ15.cpp
  Source/Tests/MISCQ7.cpp
  Source/Tests/FIRF32.cpp
  Source/Tests/FIRQ31.cpp
  Source/Tests/FIRQ15.cpp
  Source/Tests/FIRQ7.cpp
  Source/Tests/BIQUADF64.cpp
  Source/Tests/BIQUADF32.cpp
  Source/Tests/BIQUADQ31.cpp
  Source/Tests/BIQUADQ15.cpp
  Source/Tests/InterpolationTestsF32.cpp
  Source/Tests/InterpolationTestsQ31.cpp
  Source/Tests/InterpolationTestsQ15.cpp
  Source/Tests/InterpolationTestsQ7.cpp
  Source/Tests/ExampleCategoryF32.cpp
  Source/Tests/ExampleCategoryQ31.cpp
  Source/Tests/ExampleCategoryQ15.cpp
  Source/Tests/ExampleCategoryQ7.cpp
  )

if ((NOT ARMAC5) AND ((FLOAT16) OR (MVEF) OR (HELIUM) OR (NEON) OR (NEONEXPERIMENTAL)))
set(TESTSRC16 
  Source/Tests/BasicTestsF16.cpp
  Source/Tests/TransformCF16.cpp
  )
endif()

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


# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(TestingLib)
disableOptimization(FrameworkLib)


target_sources(TestingLib PRIVATE ${TESTSRC})


if(NN)
  target_sources(TestingLib PRIVATE ${NNSRC})
endif()

target_sources(TestingLib PRIVATE ${TESTSRC16})

target_sources(TestingLib PRIVATE testmain.cpp)
target_sources(TestingLib PRIVATE GeneratedSource/TestDesc.cpp)

if (BENCHMARK)
  target_compile_definitions(TestingLib PUBLIC BENCHMARK)
endif()
  

target_sources(FrameworkLib PRIVATE ${FRAMEWORKSRC})

if (EXTBENCH)
  target_compile_definitions(FrameworkLib PUBLIC EXTBENCH)
endif()

### Includes
if (REFLIB)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Source)
include(configDsp)
SET(DSP ${ROOT}/CMSIS/DSP)
target_include_directories(TestingLib PUBLIC "${DSP}/Include")
target_link_libraries(TestingLib PRIVATE "${REFLIBNAME}")
else()
target_link_libraries(TestingLib PRIVATE CMSISDSP)
endif()

if(NN)
target_link_libraries(TestingLib PRIVATE CMSISNN)
endif()
target_include_directories(TestingLib PRIVATE FrameworkInclude)
target_include_directories(TestingLib PRIVATE GeneratedInclude)

configLib(TestingLib ${ROOT})
configDsp(TestingLib ${ROOT})

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





