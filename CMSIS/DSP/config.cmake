include(CMakePrintHelpers)



include(configLib)
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/FVP)
include(configPlatform)
include(configBoot)

define_property(TARGET 
                PROPERTY DISABLEOPTIMIZATION
                 BRIEF_DOCS "Force disabling of optimizations"
                 FULL_DOCS "Force disabling of optimizations")

# Config core settings
# Configure platform (semihosting etc ...)
# May be required for some compiler
function(disableOptimization project)
  set_target_properties(${project} PROPERTIES DISABLEOPTIMIZATION ON)
endfunction()


# Config app
function (configApp project cmsisRoot)
  configcore(${project} ${cmsisRoot})
  configboot(${project} ${cmsisRoot} ${PLATFORMFOLDER})
  set_platform_core()
  core_includes(${project})
  SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)
  SET(COREID ${COREID} PARENT_SCOPE)
endfunction()