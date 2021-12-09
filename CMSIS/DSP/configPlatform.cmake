option(SEMIHOSTING "Test trace using printf" OFF)

if (PLATFORM STREQUAL "FVP")
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/FVP)
SET(PLATFORMID "FVP")
SET(PLATFORMOPT "-DFVP")
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP/Platforms/FVP)
endif()

if (PLATFORM STREQUAL "NORMALFVP")
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/NORMALFVP)
SET(PLATFORMID "NORMALFVP")
SET(PLATFORMOPT "-DNORMALFVP")
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP/Platforms/NORMALFVP)
endif()

if (PLATFORM STREQUAL "MPS3")
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/MPS3)
SET(PLATFORMID "MPS3")
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP/Platforms/MPS3)
endif()

if (PLATFORM STREQUAL "SDSIM")
SET(PLATFORMFOLDER ${SDSIMROOT})
SET(PLATFORMID "SDSIM")
list(APPEND CMAKE_MODULE_PATH ${SDSIMROOT})
endif()

if (PLATFORM STREQUAL "IPSS")
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/IPSS)
SET(PLATFORMID "IPSS")
list(APPEND CMAKE_MODULE_PATH ${ROOT}/CMSIS/DSP/Platforms/IPSS)
endif()

SET(CORE ARMCM7)


include(platform)

function(set_platform_core)

  if(EXPERIMENTAL)
     experimental_set_platform_core()
     SET(CORE ${CORE} PARENT_SCOPE) 
  endif()
  ###################
  #
  # Cortex cortex-m7
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]7([^0-9].*)?$")
    SET(CORE ARMCM7 PARENT_SCOPE)    
  endif()
  
  ###################
  #
  # Cortex cortex-m4
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]4([^0-9].*)?$")
      SET(CORE ARMCM4 PARENT_SCOPE)
  endif()
  
  ###################
  #
  # Cortex cortex-m35p
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]35([^0-9].*)?$")
      SET(CORE ARMCM35P PARENT_SCOPE)
      
  endif()
  
  ###################
  #
  # Cortex cortex-m33
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]33([^0-9].*)?$")
      SET(CORE ARMCM33 PARENT_SCOPE)
      
  endif()

  ###################
  #
  # Cortex cortex-m55
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]55([^0-9].*)?$")
    SET(CORE ARMv81MML PARENT_SCOPE)    
  endif()
  
  ###################
  #
  # Cortex cortex-m23
  #
  if (ARM_CPU  MATCHES "^[cC]ortex-[mM]23([^0-9].*)?$")
      SET(CORE ARMCM23 PARENT_SCOPE)
     
  endif()

  ###################
  #
  # Cortex cortex-m0+
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]0p([^0-9].*)?$")
      SET(CORE ARMCM0plus PARENT_SCOPE)
      
  endif()

  ###################
  #
  # Cortex cortex-m0
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]0([^0-9].*)?$")
      SET(CORE ARMCM0 PARENT_SCOPE)
      
  endif()

  ###################
  #
  # Cortex cortex-a32
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]32([^0-9].*)?$")
    SET(CORE ARMCA32 PARENT_SCOPE)
    
  endif()
  
  ###################
  #
  # Cortex cortex-a5
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]5([^0-9].*)?$")
    SET(CORE ARMCA5 PARENT_SCOPE)
    
  endif()

  ###################
  #
  # Cortex cortex-a7
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]7([^0-9].*)?$")
    SET(CORE ARMCA7 PARENT_SCOPE)
    
  endif()

  ###################
  #
  # Cortex cortex-a9
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]9([^0-9].*)?$")
    SET(CORE ARMCA9 PARENT_SCOPE)
    
  endif()

  ###################
  #
  # Cortex cortex-a15
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[aA]15([^0-9].*)?$")
    SET(CORE ARMCA15 PARENT_SCOPE)
  endif()

  ###################
  #
  # Cortex cortex-r5
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[rR]5([^0-9].*)?$")
    SET(CORE ARMCR5 PARENT_SCOPE)
  endif()

  ###################
  #
  # Cortex cortex-r8
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[rR]8([^0-9].*)?$")
    SET(CORE ARMCR8 PARENT_SCOPE)
  endif()

  ###################
  #
  # Cortex cortex-r52
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[rR]52([^0-9].*)?$")
    SET(CORE ARMCR52 PARENT_SCOPE)
  endif()

endfunction()

function(core_includes PROJECTNAME)
    if (CORTEXR)
      target_include_directories(${PROJECTNAME} PRIVATE ${CORER}/Include)
    else()
      target_include_directories(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Include)
    #target_compile_options(${PROJECTNAME} PRIVATE ${PLATFORMOPT})
  endif()
endfunction()

function (configplatformForLib PROJECTNAME ROOT)
  if (SEMIHOSTING)
    target_compile_definitions(${PROJECTNAME} PRIVATE SEMIHOSTING)
  endif()
  if (CORTEXM)
    compilerSpecificPlatformConfigLibForM(${PROJECTNAME} ${ROOT} )
  elseif(CORTEXA)
    compilerSpecificPlatformConfigLibForA(${PROJECTNAME} ${ROOT} )
  else()
    compilerSpecificPlatformConfigLibForR(${PROJECTNAME} ${ROOT} )
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
  elseif(CORTEXA)
    compilerSpecificPlatformConfigAppForA(${PROJECTNAME} ${ROOT} )
  else()
    compilerSpecificPlatformConfigAppForR(${PROJECTNAME} ${ROOT} )
  endif()

endfunction()
