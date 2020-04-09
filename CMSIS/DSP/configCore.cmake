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


  if(EXPERIMENTAL)
    experimentalConfigcore(${PROJECTNAME} ${ROOT})
    SET(COREID ${COREID} PARENT_SCOPE)
  endif()
  ###################
  #
  # CORTEX-A
  #

  # CORTEX-A15
  if (ARM_CPU  MATCHES  "^[cC]ortex-[aA]15([^0-9].*)?$" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 
  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA15 PARENT_SCOPE)
  endif()
  
  # CORTEX-A9
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]9([^0-9].*)?$" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 
  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA9 PARENT_SCOPE)
  
  endif()
  
  # CORTEX-A7
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]7([^0-9].*)?$" )
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core_A/Include")
    SET(CORTEXM OFF)
  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXA)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv7A) 

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCA7 PARENT_SCOPE)
  
  endif()
  
  # CORTEX-A5
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]5([^0-9].*)?$" )
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

  if (ARM_CPU MATCHES "^[cC]ortex-[mM]55([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMv81MML_DSP_DP_MVE_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMv81MML_DSP_DP_MVE_FP PARENT_SCOPE)    
  endif()
  
  # CORTEX-M35
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]35([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM35P_DSP_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM35P_DSP_FP PARENT_SCOPE)
  endif()
  
  # CORTEX-M33
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]33([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM33_DSP_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM33_DSP_FP PARENT_SCOPE)
  endif()
  
  # CORTEX-M23
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]23([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM23)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM23 PARENT_SCOPE)
  endif()
  
  # CORTEX-M7
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]7([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")  
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM7_DP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM7_DP PARENT_SCOPE)
  endif()
  
  # CORTEX-M4
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]4([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM4_FP)

    SET(HARDFP ON)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM4_FP PARENT_SCOPE)
  endif()
  
  # CORTEX-M3
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]3([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM3)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM3 PARENT_SCOPE)
  endif()
  
  # CORTEX-M0plus
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]0p([^0-9].*)?$")
    target_include_directories(${PROJECTNAME} PUBLIC "${ROOT}/CMSIS/Core/Include")
    target_compile_definitions(${PROJECTNAME} PUBLIC CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARMCM0P)

    SET(HARDFP OFF)
    SET(LITTLEENDIAN ON)
    SET(COREID ARMCM0P PARENT_SCOPE)
  endif()
  
  # CORTEX-M0
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]0([^0-9].*)?$")
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
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON)
  endif()
  
  if (NEONEXPERIMENTAL AND NOT CORTEXM)
    #target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON_EXPERIMENTAL __FPU_PRESENT)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_NEON_EXPERIMENTAL)
  endif()

  if (HELIUM AND CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_HELIUM)
  endif()

  if (MVEF AND CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_MVEF)
  endif()

  if (MVEI AND CORTEXM)
    target_compile_definitions(${PROJECTNAME} PRIVATE ARM_MATH_MVEI)
  endif()

  compilerSpecificCompileOptions(${PROJECTNAME} ${ROOT})

endfunction()