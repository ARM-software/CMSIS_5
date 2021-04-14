function(configure_platform PROJECTNAME ROOT CORE PLATFORMFOLDER)
    if (GCC)
      target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/Startup/GCC/startup_asm_${CORE}.S)
    # target_link_options(${PROJECTNAME} PRIVATE "-lrdimon;-lc")

    endif()
endfunction()