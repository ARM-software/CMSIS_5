include(CMakePrintHelpers)
cmake_policy(SET CMP0077 NEW)

# Config core
SET(CORTEXM ON)

option(HARDFP "Hard floating point" ON)
option(LITTLEENDIAN "Little endian" ON)
option(FASTMATHCOMPUTATIONS "Fast Math enabled" OFF)

# More detailed identification for benchmark results
SET(COREID ARMCM7)

###################
#
# ALL CORTEX
#

function(configcore PROJECTNAME ROOT)


  ###################
  #
  # CORTEX-A
  #

  # CORTEX-A15
  if (ARM_CPU STREQUAL "cortex-a15" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 
  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA15 PARENT_SCOPE)
  endif()
  
  # CORTEX-A9
  if (ARM_CPU STREQUAL "cortex-a9" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 
  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA9 PARENT_SCOPE)
  
  endif()
  
  # CORTEX-A7
  if (ARM_CPU STREQUAL "cortex-a7" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA7 PARENT_SCOPE)
  
  endif()
  
  # CORTEX-A5
  if (ARM_CPU STREQUAL "cortex-a5" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
    
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA5 PARENT_SCOPE)
  endif()

  
  ###################
  #
  # CORTEX-M
  #
  
  # CORTEX-M35
  if (ARM_CPU STREQUAL "cortex-m35")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM35P_DSP_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM35P_DSP_FP PARENT_SCOPE)
  endif()
  
  # CORTEX-M33
  if (ARM_CPU STREQUAL "cortex-m33")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM33_DSP_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM33_DSP_FP PARENT_SCOPE)
  endif()
  
  # CORTEX-M23
  if (ARM_CPU STREQUAL "cortex-m23")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM23)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM23 PARENT_SCOPE)
  endif()
  
  # CORTEX-M7
  if (ARM_CPU STREQUAL "cortex-m7")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM7_DP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM7_DP PARENT_SCOPE)
  endif()
  
  # CORTEX-M4
  if (ARM_CPU STREQUAL "cortex-m4")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM4_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM4_FP PARENT_SCOPE)
  endif()
  
  # CORTEX-M3
  if (ARM_CPU STREQUAL "cortex-m3")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM3)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM3 PARENT_SCOPE)
  endif()
  
  # CORTEX-M0plus
  if (ARM_CPU STREQUAL "cortex-m0p")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM0P)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM0P PARENT_SCOPE)
  endif()
  
  # CORTEX-M0
  if (ARM_CPU STREQUAL "cortex-m0")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM0)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM0 PARENT_SCOPE)
  endif()
  
  ###################
  #
  # FEATURES
  #
    
  if (NEON AND NOT CORTEXM)
    #target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON __FPU_PRESENT)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON)
  endif()
  
  if (NEONEXPERIMENTAL AND NOT CORTEXM)
    #target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON_EXPERIMENTAL __FPU_PRESENT)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON_EXPERIMENTAL)
  endif()

  compilerSpecificCompileOptions(${PROJECTNAME} ${ROOT})

endfunction()