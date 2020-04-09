#include(CMakePrintHelpers)
include(AddFileDependencies)

function(compilerVersion)
  execute_process(COMMAND "${CMAKE_C_COMPILER}" --version_number
     OUTPUT_VARIABLE CVERSION
     ERROR_VARIABLE CVERSION
    )
  SET(COMPILERVERSION ${CVERSION} PARENT_SCOPE)
  #cmake_print_variables(CVERSION)
  #cmake_print_variables(CMAKE_C_COMPILER)
  #MESSAGE( STATUS "CMD_OUTPUT:" ${CVERSION})
endfunction()

function(compilerSpecificCompileOptions PROJECTNAME ROOT)
  #cmake_print_properties(TARGETS ${PROJECTNAME} PROPERTIES DISABLEOPTIMIZATION)
  get_target_property(DISABLEOPTIM ${PROJECTNAME} DISABLEOPTIMIZATION)
  if ((OPTIMIZED) AND (NOT DISABLEOPTIM))
    #cmake_print_variables(DISABLEOPTIM)
    target_compile_options(${PROJECTNAME} PRIVATE "-O3")
  endif()

  if (FASTMATHCOMPUTATIONS)
      target_compile_options(${PROJECTNAME} PUBLIC "-ffast-math")
  endif()
  
  if (HARDFP)
    target_compile_options(${PROJECTNAME} PUBLIC "-mfloat-abi=hard")
  endif()
  
  if (LITTLEENDIAN)
    target_compile_options(${PROJECTNAME} PUBLIC "-mlittle-endian")
  endif()
  
  # Core specific config

  if (ARM_CPU STREQUAL "cortex-m55" )
        target_compile_options(${PROJECTNAME} PUBLIC "-fshort-enums")
        target_compile_options(${PROJECTNAME} PUBLIC "-fshort-wchar")
  endif()

  if (ARM_CPU STREQUAL "cortex-m33" )
        target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=fpv5-sp-d16")
  endif()

  if (ARM_CPU STREQUAL "cortex-m7" )
        target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=fpv5-sp-d16")
  endif()

  if (ARM_CPU STREQUAL "cortex-m4" )
        target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=fpv4-sp-d16")
  endif()
  
  if (ARM_CPU STREQUAL "cortex-a9" )
      if (NOT (NEON OR NEONEXPERIMENTAL))
        target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=vfpv3-d16-fp16")
      endif()
  endif()
  
  if (ARM_CPU STREQUAL "cortex-a7" )
      if (NOT (NEON OR NEONEXPERIMENTAL))
          target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=vfpv4-d16")
      endif()
  endif()
  
  if (ARM_CPU STREQUAL "cortex-a5" )
      if ((NEON OR NEONEXPERIMENTAL))
        target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=neon-vfpv4")
      else()
        target_compile_options(${PROJECTNAME} PUBLIC "-mfpu=vfpv4-d16")
      endif()
  endif()

  if(EXPERIMENTAL)
    experimentalCompilerSpecificCompileOptions(${PROJECTNAME} ${ROOT})
  endif()
endfunction()


function(toolchainSpecificLinkForCortexM PROJECTNAME ROOT CORE PLATFORMFOLDER HASCSTARTUP)
    # A specific library is created for ASM file
    # since we do not want standard compile flags (for C) to be applied to 
    # ASM files.
    if (HASCSTARTUP)
      target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/AC6/startup_${CORE}.c)
    else()
      target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/AC6/startup_${CORE}.s)
    endif() 
    target_include_directories(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/LinkScripts/AC6)

    set(SCATTERFILE "${PLATFORMFOLDER}/${CORE}/LinkScripts/AC6/lnk.sct")

    set_target_properties(${PROJECTNAME} PROPERTIES LINK_DEPENDS "${SCATTERFILE};${PLATFORMFOLDER}/${CORE}/LinkScripts/AC6/mem_${CORE}.h")
   
    #target_link_options(${PROJECTNAME} PRIVATE "--info=sizes")
    target_link_options(${PROJECTNAME} PRIVATE "--entry=Reset_Handler;--scatter=${SCATTERFILE}")

endfunction()

function(toolchainSpecificLinkForCortexA PROJECTNAME ROOT CORE PLATFORMFOLDER)
    target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/AC6/startup_${CORE}.c)
    

    # RTE Components.h
    target_include_directories(${PROJECTNAME} PRIVATE ${ROOT}/CMSIS/DSP/Testing)

    set(SCATTERFILE "${PLATFORMFOLDER}/${CORE}/LinkScripts/AC6/lnk.sct")

    set_target_properties(${PROJECTNAME} PROPERTIES LINK_DEPENDS "${SCATTERFILE};${PLATFORMFOLDER}/${CORE}/LinkScripts/AC6/mem_${CORE}.h")

    target_include_directories(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/LinkScripts/AC6)

    #target_link_options(${PROJECTNAME} PRIVATE "--info=sizes")
    target_link_options(${PROJECTNAME} PRIVATE "--entry=Vectors;--scatter=${SCATTERFILE}")

endfunction()

function(compilerSpecificPlatformConfigLibForM PROJECTNAME ROOT)
endfunction()

function(compilerSpecificPlatformConfigLibForA PROJECTNAME ROOT)
endfunction()

function(compilerSpecificPlatformConfigAppForM PROJECTNAME ROOT)
endfunction()

function(compilerSpecificPlatformConfigAppForA PROJECTNAME ROOT)
endfunction()