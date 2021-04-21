cmake_minimum_required (VERSION 3.14)
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
  if (CONFIGID)
     # Specific config ID (like M55 with autovectorizer)
     list(APPEND output ",${CONFIGID},")
  else()
     # Core ID is used as config ID when not specified
     list(APPEND output ",${COREID},")
  endif()
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
option(EMBEDDED "Embedded Mode" ON)

option(FLOAT16TESTS "Float16 tests" OFF)
option(ALLTESTS "All tests including Float16 tests" OFF)

option(MICROBENCH "Micro benchmarks" OFF)
option(EXTERNAL "External benchmarks or tests" OFF)
option(CACHEANALYSIS "Build with cache analysis mode enabled" OFF)

option(DISTINCT "Different generated folder for benchmarking and tests" OFF)

project(Testing)

# Needed to find the config modules


set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)


# Change behavior of configBoot for scatter file
# We use the linker files from older test framework because bigger sections are needed.
# We should migrate the linker files to this new framework.
set(TESTFRAMEWORK ON)
include(config)

if ((NOT REFLIB) AND (NOT NOCMSIS))
add_subdirectory(../Source bin_dsp)
endif()

if (NN)
add_subdirectory(${ROOT}/CMSIS/NN/Source bin_nn)
endif()

add_library(TestingLib STATIC)
add_library(FrameworkLib STATIC)

if (DISABLEFLOAT16)
    target_compile_definitions(TestingLib PRIVATE DISABLEFLOAT16) 
    target_compile_definitions(FrameworkLib PRIVATE DISABLEFLOAT16) 
endif()

if (AUTOVECTORIZE)
    target_compile_definitions(TestingLib PRIVATE ARM_MATH_AUTOVECTORIZE) 
endif()

if (CACHEANALYSIS)
    target_compile_definitions(FrameworkLib PRIVATE CACHEANALYSIS) 
endif()


if (BENCHMARK)

set(STANDARDBENCH ON)

if (MICROBENCH)
  set (MICROSRC
  Source/Benchmarks/MicroBenchmarksF32.cpp
  Source/Benchmarks/MicroBenchmarksQ31.cpp
  Source/Benchmarks/MicroBenchmarksQ15.cpp
  Source/Benchmarks/MicroBenchmarksQ7.cpp
  )
  set(STANDARDBENCH OFF)

endif()

if (EXTERNAL)
  add_subdirectory(${EXTERNALDIR} bin_external)
  set(STANDARDBENCH OFF)
endif()

set (NNSRC
  Source/Benchmarks/FullyConnectedBench.cpp
  Source/Benchmarks/PoolingBench.cpp
  )

 if (STANDARDBENCH)
 set(TESTSRC 
   Source/Benchmarks/BasicMathsBenchmarksF32.cpp
   Source/Benchmarks/BasicMathsBenchmarksQ31.cpp
   Source/Benchmarks/BasicMathsBenchmarksQ15.cpp
   Source/Benchmarks/BasicMathsBenchmarksQ7.cpp
   Source/Benchmarks/ComplexMathsBenchmarksF32.cpp
   Source/Benchmarks/ComplexMathsBenchmarksQ31.cpp
   Source/Benchmarks/ComplexMathsBenchmarksQ15.cpp
   Source/Benchmarks/QuaternionMathsBenchmarksF32.cpp
   Source/Benchmarks/BayesF32.cpp
   Source/Benchmarks/SVMF32.cpp
   Source/Benchmarks/DistanceF32.cpp
   Source/Benchmarks/DistanceU32.cpp
   Source/Benchmarks/StatsF64.cpp
   Source/Benchmarks/StatsF32.cpp
   Source/Benchmarks/StatsQ31.cpp
   Source/Benchmarks/StatsQ15.cpp
   Source/Benchmarks/StatsQ7.cpp
   Source/Benchmarks/FIRF32.cpp
   Source/Benchmarks/FIRQ31.cpp
   Source/Benchmarks/FIRQ15.cpp
   Source/Benchmarks/FIRQ7.cpp
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
   Source/Benchmarks/UnaryQ7.cpp
   Source/Benchmarks/BinaryF32.cpp
   Source/Benchmarks/BinaryQ31.cpp
   Source/Benchmarks/BinaryQ15.cpp
   Source/Benchmarks/BinaryQ7.cpp
   Source/Benchmarks/TransformF32.cpp
   Source/Benchmarks/TransformQ31.cpp
   Source/Benchmarks/TransformQ15.cpp
   )
 target_include_directories(TestingLib PRIVATE Include/Benchmarks)
 endif()

 if ((NOT ARMAC5) AND (FLOAT16TESTS OR ALLTESTS) AND ((FLOAT16) OR (MVEF) OR (HELIUM) OR (NEON) OR (NEONEXPERIMENTAL)))
 set(TESTSRC16 Source/Benchmarks/BasicMathsBenchmarksF16.cpp
   Source/Benchmarks/ComplexMathsBenchmarksF16.cpp
   Source/Benchmarks/BayesF16.cpp
   Source/Benchmarks/SVMF16.cpp
   Source/Benchmarks/DistanceF16.cpp
   Source/Benchmarks/StatsF16.cpp
   Source/Benchmarks/FIRF16.cpp
   Source/Benchmarks/MISCF16.cpp
   Source/Benchmarks/BIQUADF16.cpp
   Source/Benchmarks/FastMathF16.cpp
   Source/Benchmarks/SupportF16.cpp
   Source/Benchmarks/SupportBarF16.cpp
   Source/Benchmarks/UnaryF16.cpp
   Source/Benchmarks/BinaryF16.cpp
   Source/Benchmarks/TransformF16.cpp
  )
endif()

else()

set(STANDARDTEST ON)

set(NNSRC
  Source/Tests/NNSupport.cpp
  Source/Tests/Pooling.cpp
  Source/Tests/Softmax.cpp
  Source/Tests/FullyConnected.cpp
  )

if (EXTERNAL)
  add_subdirectory(${EXTERNALDIR} bin_external)
  set(STANDARDTEST OFF)
endif()

if (STANDARDTEST)

if (BASICMATH)
  set(BASICMATHSRC Source/Tests/BasicTestsF32.cpp
  Source/Tests/BasicTestsQ31.cpp
  Source/Tests/BasicTestsQ15.cpp
  Source/Tests/BasicTestsQ7.cpp)
endif()

if (QUATERNIONMATH)
  set(QUATERNIONMATHSRC Source/Tests/QuaternionTestsF32.cpp
  )
endif()

if (COMPLEXMATH)
  set(COMPLEXMATHSRC Source/Tests/ComplexTestsF32.cpp
  Source/Tests/ComplexTestsQ31.cpp
  Source/Tests/ComplexTestsQ15.cpp)
endif()

if (CONTROLLER)
  set(CONTROLLERSRC )
endif()

if (FASTMATH)
  set(FASTMATHSRC Source/Tests/FastMathF32.cpp
  Source/Tests/FastMathQ31.cpp
  Source/Tests/FastMathQ15.cpp)
endif()

if (FILTERING)
  set(FILTERINGSRC Source/Tests/DECIMF32.cpp
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
  Source/Tests/BIQUADQ15.cpp)
endif()

if (MATRIX)
  set(MATRIXSRC  Source/Tests/UnaryTestsQ31.cpp
  Source/Tests/UnaryTestsQ15.cpp
  Source/Tests/UnaryTestsQ7.cpp
  Source/Tests/UnaryTestsF32.cpp
  Source/Tests/UnaryTestsF64.cpp
  Source/Tests/BinaryTestsF32.cpp
  Source/Tests/BinaryTestsF64.cpp
  Source/Tests/BinaryTestsQ31.cpp
  Source/Tests/BinaryTestsQ15.cpp
  Source/Tests/BinaryTestsQ7.cpp)
endif()

if (STATISTICS)
  set(STATISTICSSRC Source/Tests/StatsTestsF32.cpp
  Source/Tests/StatsTestsF64.cpp
  Source/Tests/StatsTestsQ31.cpp
  Source/Tests/StatsTestsQ15.cpp
  Source/Tests/StatsTestsQ7.cpp)
endif()

if (SUPPORT)
  set(SUPPORTSRC Source/Tests/SupportTestsF32.cpp
  Source/Tests/SupportTestsQ31.cpp
  Source/Tests/SupportTestsQ15.cpp
  Source/Tests/SupportTestsQ7.cpp
  Source/Tests/SupportBarTestsF32.cpp)
endif()

if (TRANSFORM)
  set(TRANSFORMSRC Source/Tests/TransformCF64.cpp
  Source/Tests/TransformCF32.cpp
  Source/Tests/TransformRF64.cpp
  Source/Tests/TransformRF32.cpp
  Source/Tests/TransformCQ31.cpp
  Source/Tests/TransformRQ31.cpp
  Source/Tests/TransformCQ15.cpp
  Source/Tests/TransformRQ15.cpp)
endif()

if (SVM)
  set(SVMSRC Source/Tests/SVMF32.cpp)
endif()

if (BAYES)
  set(BAYESSRC Source/Tests/BayesF32.cpp)
endif()

if (DISTANCE)
  set(DISTANCESRC Source/Tests/DistanceTestsF32.cpp
  Source/Tests/DistanceTestsU32.cpp)
endif()

if (INTERPOLATION)
  set(INTERPOLATIONSRC Source/Tests/InterpolationTestsF32.cpp
     Source/Tests/InterpolationTestsQ31.cpp
     Source/Tests/InterpolationTestsQ15.cpp
     Source/Tests/InterpolationTestsQ7.cpp)
endif()

set(TESTSRC 
     ${BASICMATHSRC}
     ${COMPLEXMATHSRC}
     ${CONTROLLERSRC}
     ${FASTMATHSRC}
     ${FILTERINGSRC}
     ${MATRIXSRC}
     ${STATISTICSSRC}
     ${SUPPORTSRC}
     ${TRANSFORMSRC}
     ${SVMSRC}
     ${BAYESSRC}
     ${DISTANCESRC}
     ${QUATERNIONMATHSRC}
     ${INTERPOLATIONSRC}
     #Source/Tests/ExampleCategoryF32.cpp
     #Source/Tests/ExampleCategoryQ31.cpp
     #Source/Tests/ExampleCategoryQ15.cpp
     #Source/Tests/ExampleCategoryQ7.cpp
  )


if ((NOT ARMAC5) AND (FLOAT16TESTS OR ALLTESTS) AND ((FLOAT16) OR (MVEF) OR (HELIUM) OR (NEON) OR (NEONEXPERIMENTAL)))
set(TESTSRC16 
  Source/Tests/BasicTestsF16.cpp
  Source/Tests/ComplexTestsF16.cpp
  Source/Tests/InterpolationTestsF16.cpp
  Source/Tests/StatsTestsF16.cpp
  Source/Tests/FIRF16.cpp
  Source/Tests/BIQUADF16.cpp
  Source/Tests/MISCF16.cpp
  Source/Tests/BinaryTestsF16.cpp
  Source/Tests/UnaryTestsF16.cpp
  Source/Tests/TransformCF16.cpp
  Source/Tests/TransformRF16.cpp
  Source/Tests/SupportTestsF16.cpp
  Source/Tests/SupportBarTestsF16.cpp
  Source/Tests/FastMathF16.cpp
  Source/Tests/DistanceTestsF16.cpp
  Source/Tests/SVMF16.cpp
  Source/Tests/BayesF16.cpp
  )
endif()
endif() 

target_include_directories(TestingLib PUBLIC Include/Tests)

endif()

set(FRAMEWORKSRC
  FrameworkSource/Test.cpp
  FrameworkSource/IORunner.cpp
  FrameworkSource/ArrayMemory.cpp
  FrameworkSource/Pattern.cpp
  FrameworkSource/PatternMgr.cpp
  FrameworkSource/Error.cpp
  FrameworkSource/Timing.cpp
  FrameworkSource/Generators.cpp
  FrameworkSource/Calibrate.cpp
  )

if (EMBEDDED)
  set(FRAMEWORKMODESRC FrameworkSource/FPGA.cpp)
else()
  set(FRAMEWORKMODESRC FrameworkSource/Semihosting.cpp)
endif()


# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(TestingLib)
disableOptimization(FrameworkLib)

## Only build f16 version when running float16tests or all tests
## and float16 are supported
if (ARMAC5)
  target_sources(TestingLib PRIVATE ${TESTSRC})
elseif ((FLOAT16) OR (MVEF) OR (HELIUM) OR (NEON) OR (NEONEXPERIMENTAL))
      if (ALLTESTS)
        target_sources(TestingLib PRIVATE ${TESTSRC16})
        target_sources(TestingLib PRIVATE ${TESTSRC})
      elseif (FLOAT16TESTS)
        target_sources(TestingLib PRIVATE ${TESTSRC16})
      else()
        target_sources(TestingLib PRIVATE ${TESTSRC})
      endif()
else()
      target_sources(TestingLib PRIVATE ${TESTSRC})
endif()


if(NN)
  target_sources(TestingLib PRIVATE ${NNSRC})
endif()



target_sources(TestingLib PRIVATE testmain.cpp)

if ((DISTINCT) AND (BENCHMARK))
target_sources(TestingLib PRIVATE GeneratedSourceBench/TestDesc.cpp)
else()
target_sources(TestingLib PRIVATE GeneratedSource/TestDesc.cpp)
endif()

if (EMBEDDED)
    target_compile_definitions(TestingLib PUBLIC EMBEDDED)
endif()

if (BENCHMARK)
  target_compile_definitions(TestingLib PUBLIC BENCHMARK)
  if (MICROBENCH)
    target_sources(TestingLib PRIVATE ${MICROSRC})
  endif()
endif()
  

target_sources(FrameworkLib PRIVATE ${FRAMEWORKSRC})
target_sources(FrameworkLib PRIVATE ${FRAMEWORKMODESRC})

target_compile_definitions(FrameworkLib PRIVATE ${PLATFORMOPT}) 

if (DISABLEFLOAT16)
    target_compile_definitions(FrameworkLib PRIVATE DISABLEFLOAT16) 
endif()

if (BENCHMARK)
    target_compile_definitions(FrameworkLib PUBLIC BENCHMARK)
endif()

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
if (NOT NOCMSIS)
target_link_libraries(TestingLib PRIVATE CMSISDSP)
endif()
endif()

if(NN)
target_link_libraries(TestingLib PRIVATE CMSISNN)
endif()
target_include_directories(TestingLib PRIVATE FrameworkInclude)

if ((DISTINCT) AND (BENCHMARK))
target_include_directories(TestingLib PRIVATE GeneratedIncludeBench)
else()
target_include_directories(TestingLib PRIVATE GeneratedInclude)
endif()

configLib(TestingLib ${ROOT})
#configDsp(TestingLib ${ROOT})

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
# To see the file in the scatter load, it must not because
# linked in a .a archive
if ((DISTINCT) AND (BENCHMARK))
target_include_directories(Testing PRIVATE GeneratedIncludeBench)
else()
target_include_directories(Testing PRIVATE GeneratedInclude)
endif()

target_sources(Testing PRIVATE patterndata.c)

# With -O2, generated code is crashing on some cycle accurate models.
# (cpp part)
disableOptimization(Testing)

configApp(Testing ${ROOT})

target_link_libraries(Testing PRIVATE TestingLib)
target_link_libraries(Testing PRIVATE FrameworkLib)

if (EXTERNAL)
  target_include_directories(${EXTERNALPROJECT} PRIVATE FrameworkInclude)
  if ((DISTINCT) AND (BENCHMARK))
    target_include_directories(${EXTERNALPROJECT} PRIVATE GeneratedIncludeBench)
  else()
    target_include_directories(${EXTERNALPROJECT} PRIVATE GeneratedInclude)
  endif()
  target_link_libraries(TestingLib PRIVATE ${EXTERNALPROJECT})
endif()

writeConfig(${CMAKE_CURRENT_BINARY_DIR}/currentConfig.csv)


find_package(Doxygen)

if(DOXYGEN_FOUND)

  # exclude sqlite code
  set(DOXYGEN_EXCLUDE_PATTERNS
        */sqlite3/*
  )
  # doxygen settings can be set here, prefixed with "DOXYGEN_"
  set(DOXYGEN_SOURCE_BROWSER NO)
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_EXTRACT_PRIVATE NO)
  set(DOXYGEN_GENERATE_XML NO)
  set(DOXYGEN_OPTIMIZE_OUTPUT_FOR_C YES)
  set(DOXYGEN_EXTRACT_LOCAL_CLASSES NO)
  set(DOXYGEN_TYPEDEF_HIDES_STRUCT YES)
  set(DOXYGEN_ENABLE_PREPROCESSING YES)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
  set(DOXYGEN_MARKDOWN_SUPPORT YES)
  set(DOXYGEN_AUTOLINK_SUPPORT YES)
  set(DOXYGEN_QUIET YES)
  set(DOXYGEN_SHOW_USED_FILES NO)
  set(DOXYGENEXTRACT_LOCAL_CLASSES NO)
  set(DOXYGENSHOW_INCLUDE_FILES NO)
  set(DOXYGEN_TYPEDEF_HIDES_STRUCT YES)
  set(DOXYGEN_FULL_PATH_NAMES NO)
  set(DOXYGEN_RECURSIVE YES)
  set(DOXYGEN_REFERENCES_LINK_SOURCE NO)
  set(DOXYGEN_EXCLUDE_PATTERNS "*/RTE/*")
  set(DOXYGEN_PREDEFINED "__ARM_FP16_FORMAT_IEEE=1;ARM_FLOAT16_SUPPORTED=1;__ALIGNED(x)=")
  set(DOXYGEN_EXAMPLE_PATH "${ROOT}/CMSIS/DSP/Examples/ARM")
  set(DOXYGEN_IMAGE_PATH "${ROOT}/CMSIS/DoxyGen/DSP/src/images")
  set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
  set(DOXYGEN_LAYOUT_FILE "${ROOT}/CMSIS/DoxyGen/Doxygen_Templates/DoxygenLayout_forUser.xml")
  # this target will only be built if specifically asked to.
  # run "make api-docs" to create the doxygen documentation
  doxygen_add_docs(
    docs
    ${ROOT}/CMSIS/DSP/Source
    ${ROOT}/CMSIS/DSP/Examples/ARM
    ${ROOT}/CMSIS/DSP/Include
    ${ROOT}/CMSIS/DoxyGen/DSP/src/history.txt
    COMMENT "Generate API-documents"
  )

endif(DOXYGEN_FOUND)


