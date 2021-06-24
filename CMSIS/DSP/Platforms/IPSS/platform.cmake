function(configure_platform PROJECTNAME ROOT CORE PLATFORMFOLDER)
    #if (${CORE} STREQUAL "ARMCA32")
    #    target_sources(${PROJECTNAME} PRIVATE ${PLATFORMFOLDER}/${CORE}/pagetables.s)
    #    
    #endif()
    target_include_directories(${PROJECTNAME} PUBLIC ${PLATFORMFOLDER})
endfunction()