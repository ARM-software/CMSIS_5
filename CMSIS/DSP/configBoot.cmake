include(CMakePrintHelpers)



get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

cmake_print_variables(PROJECT_NAME)


function(cortexm CORE PROJECT_NAME ROOT PLATFORMFOLDER CSTARTUP)
   
    target_include_directories(${PROJECT_NAME} PRIVATE ${ROOT}/CMSIS/Core/Include)
    
    target_sources(${PROJECT_NAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/system_${CORE}.c)
    

    toolchainSpecificLinkForCortexM(${PROJECT_NAME} ${ROOT} ${CORE} ${PLATFORMFOLDER} ${CSTARTUP})

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

  if(EXPERIMENTAL)
    experimentalConfigboot(${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    if (ISCORTEXM)
      cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER} ${HASCSTARTUP})    
    else()
      cortexa(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    endif()
  endif()
  ###################
  #
  # Cortex M
  #
  # C startup for M55 boot code
  if (ARM_CPU MATCHES "^[cC]ortex-[mM]55([^0-9].*)?$")
    cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER} ON)    
  elseif (ARM_CPU MATCHES  "^[cC]ortex-[Mm].*$")
    cortexm(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER} OFF)    
  endif()
  
  
  ###################
  #
  # Cortex cortex-a5
  #
  if (ARM_CPU MATCHES "^[cC]ortex-[Aa].*")
    cortexa(${CORE} ${PROJECT_NAME} ${ROOT} ${PLATFORMFOLDER})
    
  endif()

  SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)
  
endfunction()

