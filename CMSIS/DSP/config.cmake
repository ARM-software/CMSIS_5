include(CMakePrintHelpers)



include(configLib)
SET(PLATFORMFOLDER ${ROOT}/CMSIS/DSP/Platforms/FVP)
include(configPlatform)
include(configBoot)

define_property(TARGET 
                PROPERTY DISABLEOPTIMIZATION
                 BRIEF_DOCS "Force disabling of optimizations"
                 FULL_DOCS "Force disabling of optimizations")

define_property(TARGET 
                PROPERTY DISABLEHALFFLOATSUPPORT
                 BRIEF_DOCS "Force disabling of f16 support"
                 FULL_DOCS "Force disabling of f16 support")

# Config core settings
# Configure platform (semihosting etc ...)
# May be required for some compiler
function(disableOptimization project)
  set_target_properties(${project} PROPERTIES DISABLEOPTIMIZATION ON)
endfunction()

function(disableHasfFloat project)
  set_target_properties(${project} PROPERTIES DISABLEHALFFLOATSUPPORT ON)
endfunction()


# Config app
function (configApp project cmsisRoot)
  # When the C compiler is used to process ASM files, the fp16 option 
  # is not always recognized.
  # So, FP16 option is ignored when building boot code 
  # which is containing ASM
  disableHasfFloat(${project})
  configcore(${project} ${cmsisRoot})
  configboot(${project} ${cmsisRoot} ${PLATFORMFOLDER})
  set_platform_core()
  core_includes(${project})
  SET(PLATFORMID ${PLATFORMID} PARENT_SCOPE)
  SET(COREID ${COREID} PARENT_SCOPE)
endfunction()