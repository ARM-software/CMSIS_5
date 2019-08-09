option(SEMIHOSTING "Test trace using printf" ON)

if (PLATFORM STREQUAL "FVP")
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/FVP)
SET(PLATFORMID "FVP")
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP/Platforms/FVP)
endif()

if (PLATFORM STREQUAL "SDSIM")
SET(PLATFORMFOLDER ${SDSIMROOT})
SET(PLATFORMID "SDSIM")
list(APPEND CMAKE_MODULE_PATH ${SDSIMROOT})
endif()

SET(CORE ARMCM7)

include(platform)

function(set_platform_core)
  ###################
  #
  # Cortex cortex-m7
  #
  if (ARM_CPU STREQUAL "cortex-m7")
    SET(CORE ARMCM7 PARENT_SCOPE)    
  endif()
  
  ###################
  #
  # Cortex cortex-m4
  #
  if (ARM_CPU STREQUAL "cortex-m4")
      SET(CORE ARMCM4 PARENT_SCOPE)
  endif()
  
  ###################
  #
  # Cortex cortex-m35p
  #
  if (ARM_CPU STREQUAL "cortex-m35")
      SET(CORE ARMCM35P PARENT_SCOPE)
      
  endif()
  
  ###################
  #
  # Cortex cortex-m33
  #
  if (ARM_CPU STREQUAL "cortex-m33")
      SET(CORE ARMCM33 PARENT_SCOPE)
      
  endif()
  
  ###################
  #
  # Cortex cortex-m23
  #
  if (ARM_CPU STREQUAL "cortex-m23")
      SET(CORE ARMCM23 PARENT_SCOPE)
     
  endif()

  ###################
  #
  # Cortex cortex-m0+
  #
  if (ARM_CPU STREQUAL "cortex-m0p")
      SET(CORE ARMCM0plus PARENT_SCOPE)
      
  endif()

  ###################
  #
  # Cortex cortex-m0
  #
  if (ARM_CPU STREQUAL "cortex-m0")
      SET(CORE ARMCM0 PARENT_SCOPE)
      
  endif()
  
  ###################
  #
  # Cortex cortex-a5
  #
  if (ARM_CPU STREQUAL "cortex-a5")
    SET(CORE ARMCA5 PARENT_SCOPE)
    
  endif()

  ###################
  #
  # Cortex cortex-a7
  #
  if (ARM_CPU STREQUAL "cortex-a7")
    SET(CORE ARMCA7 PARENT_SCOPE)
    
  endif()

  ###################
  #
  # Cortex cortex-a9
  #
  if (ARM_CPU STREQUAL "cortex-a9")
    SET(CORE ARMCA9 PARENT_SCOPE)
    
  endif()

  ###################
  #
  # Cortex cortex-a15
  #
  if (ARM_CPU STREQUAL "cortex-a15")
    SET(CORE ARMCA15 PARENT_SCOPE)
  endif()
endfunction()

function(core_includes PROJECTNAME)
    target_include_directories(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Include)
endfunction()

function (configplatformForLib PROJECTNAME ROOT)
  if (SEMIHOSTING)
    target_compile_definitions(${PROJECTNAME} PRIVATE SEMIHOSTING)
  endif()
  if (CORTEXM)
    compilerSpecificPlatformConfigLibForM(${PROJECTNAME} ${ROOT} )
  else()
    compilerSpecificPlatformConfigLibForA(${PROJECTNAME} ${ROOT} )
  endif()

endfunction()

function (configplatformForApp PROJECTNAME ROOT CORE PLATFORMFOLDER)
  if (SEMIHOSTING)
    target_compile_definitions(${PROJECTNAME} PRIVATE SEMIHOSTING)
  endif()

  configure_platform(${PROJECTNAME} ${ROOT} ${CORE} ${PLATFORMFOLDER})
  SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)
  
  if (CORTEXM)
    compilerSpecificPlatformConfigAppForM(${PROJECTNAME} ${ROOT} )
  else()
    compilerSpecificPlatformConfigAppForA(${PROJECTNAME} ${ROOT} )
  endif()

endfunction()
