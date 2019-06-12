include(CMakePrintHelpers)

enable_language(CXX C ASM)


# Otherwise there is a .obj on windows and it creates problems
# with armlink. 
SET(CMAKE_C_OUTPUT_EXTENSION .o)
SET(CMAKE_CXX_OUTPUT_EXTENSION .o)
SET(CMAKE_ASM_OUTPUT_EXTENSION .o)


get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

cmake_print_variables(PROJECT_NAME)


function(cortexm CORE PROJECT_NAME ROOT PLATFORMFOLDER)
   
    target_include_directories(${PROJECT_NAME} PRIVATE ${ROOT}/CMSIS/Core/Include)
    
    target_sources(${PROJECT_NAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/system_${CORE}.c)
    

    toolchainSpecificLinkForCortexM(${PROJECT_NAME} ${ROOT} ${CORE} ${PLATFORMFOLDER})

    configplatformForApp(${PROJECT_NAME} ${ROOT} ${CORE} ${PLATFORMFOLDER})
    SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)

endfunction()

function(cortexa CORE PROJECT_NAME ROOT PLATFORMFOLDER)
    target_include_directories(${PROJECT_NAME} PRIVATE ${ROOT}/CMSIS/Core_A/Include)

    target_sources(${PROJECT_NAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/irq_ctrl_gic.c)
    target_sources(${PROJECT_NAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/mmu_${CORE}.c)
    target_sources(${PROJECT_NAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/system_${CORE}.c)

    
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DCMSIS_device_header="${CORE}.h")

    toolchainSpecificLinkForCortexA(${PROJECT_NAME} ${ROOT} ${CORE} ${PLATFORMFOLDER})

    configplatformForApp(${PROJECT_NAME} ${ROOT} ${CORE} ${PLATFORMFOLDER})
    SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)
endfunction()

function(configboot PROJECT_NAME ROOT PLATFORMFOLDER)

  target_include_directories(${PROJECT_NAME} PRIVATE ${ROOT}/CMSIS/DSP/Include)
  set_platform_core()
  ###################
  #
  # Cortex cortex-m7
  #
  if (ARM_CPU STREQUAL "cortex-m7")
    cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})    
    
  endif()
  
  ###################
  #
  # Cortex cortex-m4
  #
  if (ARM_CPU STREQUAL "cortex-m4")
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
  endif()
  
  ###################
  #
  # Cortex cortex-m35p
  #
  if (ARM_CPU STREQUAL "cortex-m35")
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
      
  endif()
  
  ###################
  #
  # Cortex cortex-m33
  #
  if (ARM_CPU STREQUAL "cortex-m33")
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
      
  endif()
  
  ###################
  #
  # Cortex cortex-m23
  #
  if (ARM_CPU STREQUAL "cortex-m23")
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
     
  endif()

  ###################
  #
  # Cortex cortex-m0+
  #
  if (ARM_CPU STREQUAL "cortex-m0p")
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
      
  endif()

  ###################
  #
  # Cortex cortex-m0
  #
  if (ARM_CPU STREQUAL "cortex-m0")
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
      
  endif()
  
  ###################
  #
  # Cortex cortex-a5
  #
  if (ARM_CPU STREQUAL "cortex-a5")
    cortexa(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    
  endif()

  ###################
  #
  # Cortex cortex-a7
  #
  if (ARM_CPU STREQUAL "cortex-a7")
    cortexa(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    
  endif()

  ###################
  #
  # Cortex cortex-a9
  #
  if (ARM_CPU STREQUAL "cortex-a9")
    cortexa(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    
  endif()

  ###################
  #
  # Cortex cortex-a15
  #
  if (ARM_CPU STREQUAL "cortex-a15")
    cortexa(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    
  endif()

  SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)
  
endfunction()

