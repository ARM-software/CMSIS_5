cmake_minimum_required (VERSION 3.6)

project(CMSISDSPComplexMath)

include(configLib)
include(configDsp)

file(GLOB SRC "./*_*.c")

add_library(CMSISDSPComplexMath STATIC ${SRC})

configLib(CMSISDSPComplexMath ${ROOT})
configDsp(CMSISDSPComplexMath ${ROOT})

### Includes
target_include_directories(CMSISDSPComplexMath PUBLIC "${DSP}/Include")



