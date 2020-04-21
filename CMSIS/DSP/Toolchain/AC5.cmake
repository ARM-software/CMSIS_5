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
  get_target_property(DISABLEHALF ${PROJECTNAME} DISABLEHALFFLOATSUPPORT)

  #cmake_print_variables(${PROJECTNAME} DISABLEHALF DISABLEOPTIM)
  # Add support for the type __fp16 even if there is no HW
  # support for it. But support disabled when building boot code
  if ((NOT DISABLEHALF) AND (FLOAT16))
  target_compile_options(${PROJECTNAME} PRIVATE "--fp16_format=alternative")
  endif()
  
  if ((OPTIMIZED) AND (NOT DISABLEOPTIM))
    #cmake_print_variables(DISABLEOPTIM)
    target_compile_options(${PROJECTNAME} PRIVATE "-O2")
  endif()

  if (FASTMATHCOMPUTATIONS)
      target_compile_options(${PROJECTNAME} PUBLIC "-ffast-math")
  endif()
  
  #if (HARDFP)
  #  target_compile_options(${PROJECTNAME} PUBLIC "-mfloat-abi=hard")
  #endif()
  
  #if (LITTLEENDIAN)
  #  target_compile_options(${PROJECTNAME} PUBLIC "-mlittle-endian")
  #endif()

   if (ARM_CPU STREQUAL "Cortex-M7.fp.dp" )
        target_compile_options(${PROJECTNAME} PUBLIC "--fpu=FPv5_D16")
        target_compile_options(${PROJECTNAME} PUBLIC "--thumb")
  endif()

  if (ARM_CPU STREQUAL "Cortex-A5.neon" )
        target_compile_options(${PROJECTNAME} PUBLIC "--fp16_format=ieee")
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
      target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/AC5/startup_${CORE}.c)
    else()
      target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/AC5/startup_${CORE}.s)
    endif() 
    target_include_directories(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/LinkScripts/AC5)

    set(SCATTERFILE "${PLATFORMFOLDER}/${CORE}/LinkScripts/AC5/lnk.sct")

    set_target_properties(${PROJECTNAME} PROPERTIES LINK_DEPENDS "${SCATTERFILE};${PLATFORMFOLDER}/${CORE}/LinkScripts/AC5/mem_${CORE}.h")
   
    #target_link_options(${PROJECTNAME} PRIVATE "--info=sizes")
    target_link_options(${PROJECTNAME} PRIVATE "--entry=Reset_Handler;--scatter=${SCATTERFILE}")

endfunction()

function(toolchainSpecificLinkForCortexA PROJECTNAME ROOT CORE PLATFORMFOLDER)
    target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/AC5/startup_${CORE}.c)
    

    # RTE Components.h
    target_include_directories(${PROJECTNAME} PRIVATE ${ROOT}/CMSIS/DSP/Testing)

    set(SCATTERFILE "${PLATFORMFOLDER}/${CORE}/LinkScripts/AC5/lnk.sct")

    set_target_properties(${PROJECTNAME} PROPERTIES LINK_DEPENDS "${SCATTERFILE};${PLATFORMFOLDER}/${CORE}/LinkScripts/AC5/mem_${CORE}.h")

    target_include_directories(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/LinkScripts/AC5)

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