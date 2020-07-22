cmake_minimum_required (VERSION 3.14)

project(CMSISDSPInterpolation)

include(configLib)
include(configDsp)

file(GLOB SRC "./*_*.c")

add_library(CMSISDSPInterpolation STATIC ${SRC})

configLib(CMSISDSPInterpolation ${ROOT})
configDsp(CMSISDSPInterpolation ${ROOT})

### Includes
target_include_directories(CMSISDSPInterpolation PUBLIC "${DSP}/Include")



