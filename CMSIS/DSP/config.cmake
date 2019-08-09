include(Toolchain/Tools)
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/FVP)
option(OPTIMIZED "Compile for speed" ON)
include(configPlatform)
include(configBoot)
include(configCore)

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


function(configLib project cmsisRoot)
  configcore(${project} ${cmsisRoot})
  configplatformForLib(${project} ${cmsisRoot})
  SET(COREID ${COREID} PARENT_SCOPE)
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